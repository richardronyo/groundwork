"""
Groundwork — Stage 3: Business Rules → PostgreSQL (OpenAI)

Reads the file list from PostgreSQL (populated by metadata.py), extracts
business rules as user stories via OpenAI, and writes them to the
business_rules table. Marks each file rules_extracted = TRUE.

Resumable: with --only-unprocessed it skips files already done, so an
interrupted run can be continued without redoing work.

Usage:
    python3 extract_business_rules.py --repo flask --repo-path ./flask
    python3 extract_business_rules.py --repo flask --repo-path ./flask --only-unprocessed
    python3 extract_business_rules.py --repo flask --repo-path ./flask --no-synthesize

Requirements:
    pip install openai psycopg python-dotenv
"""

import json
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection, save_business_rules, save_key_points,
    get_files, load_business_rules_from_db, list_repositories,
)

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL           = "gpt-5-mini"
SYNTHESIS_MODEL = "gpt-5"
MAX_FILE_LINES  = 80
RETRY_DELAY     = 2

CODE_LANGUAGES = {
    "Python", "JavaScript", "TypeScript",
    "JavaScript (React)", "TypeScript (React)",
    "Java", "Kotlin", "C#", "C++", "C", "C/C++ Header",
    "Go", "Rust", "Ruby", "PHP", "Swift", "Shell", "Batch", "SQL",
}

FILE_RULES_SYSTEM = "You extract business rules from code."

FILE_RULES_PROMPT = """You are a software analyst.
Analyze the following parsed file JSON and extract all business rules.
Focus only on domain logic, validation rules, constraints, workflows, and decision logic.
Return the business rules as a JSON array of strings.
Each string should be a user story in the format:
"As a [role], I want [action] so that [benefit]."
where the role is typically "store owner" or "customer" depending on context,
the action describes what the system should do, and the benefit explains why.
Write this as natural, readable English.

If the file has no meaningful business logic, return an empty array [].
Respond ONLY with a JSON array of strings. No preamble, no markdown fences, no explanation.

File: {filename}
Language: {language}

Parsed File JSON:
----------------------
{file_json}
"""

# Hard cap on the number of repo-level key points. This is also the ChromaDB
# vector dimension (one dimension per key point), so keeping it small keeps
# embedding fast and retrieval meaningful.
MAX_KEY_POINTS = 15

SYNTHESIS_PROMPT = """\
Your task is to produce a repository capability catalog.

Identify the {max_points} MOST important, distinct capabilities of this codebase.
For each:
- create one capability statement
- keep it specific
- avoid architectural marketing language
- preserve constraints and conditions
- merge overlapping behaviors rather than repeating them

Output AT MOST {max_points} capabilities. Fewer is fine if the codebase is small.
Never output more than {max_points}.

Respond ONLY with a JSON array of strings.
No markdown. No explanations. No headings.

Business Rules:
{rules_block}
"""


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in .env or environment.")
        sys.exit(1)
    return OpenAI(api_key=OPENAI_API_KEY)


def call_openai(client, prompt, model, system=None, retries=3) -> str:
    for attempt in range(retries):
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error: {e} — retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def parse_json_response(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def read_file(path: Path, max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"... ({max_lines} lines shown, file continues)")
                    break
                lines.append(line)
        return "".join(lines)
    except OSError:
        return ""


# Hard token ceiling per synthesis request. Each batch prompt (rules + scaffold)
# plus the model's output must stay under the model's TPM limit. We budget the
# INPUT rules to a fraction of the ceiling so the prompt wrapper and the response
# still fit under 500k.
SYNTH_MAX_TOKENS   = 500_000          # absolute ceiling per request (API limit)
SYNTH_OUTPUT_BUDGET = 32_000          # reserve for the model's output
SYNTH_SCAFFOLD_BUDGET = 2_000         # reserve for the prompt template text
# Safety headroom. When tiktoken is unavailable we count tokens with a rough
# estimate that can UNDERcount, so we keep a wide margin below the hard ceiling.
SYNTH_SAFETY_TOKENS = 200_000
# Tokens of actual rules text we allow per batch (well under 500k):
SYNTH_INPUT_BUDGET = (SYNTH_MAX_TOKENS - SYNTH_OUTPUT_BUDGET
                      - SYNTH_SCAFFOLD_BUDGET - SYNTH_SAFETY_TOKENS)


_ENCODER_CACHE = "unset"

def _get_encoder():
    """
    Returns a tiktoken encoder, or None if tiktoken is missing OR its vocab can't
    be loaded (e.g. offline). Cached so we only try once. On None, callers fall
    back to a conservative char-based estimate.
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE != "unset":
        return _ENCODER_CACHE
    try:
        import tiktoken
        try:
            _ENCODER_CACHE = tiktoken.encoding_for_model(SYNTHESIS_MODEL)
        except Exception:
            _ENCODER_CACHE = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # ImportError, or vocab download failure, or anything else — degrade safely
        _ENCODER_CACHE = None
    return _ENCODER_CACHE


def count_tokens(text: str, encoder=None) -> int:
    """Exact token count via tiktoken; falls back to a ~4-chars/token estimate."""
    enc = encoder or _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return (len(text) + 2) // 3  # conservative estimate (~3 chars/token)

BATCH_SUMMARY_PROMPT = """\
Below are business rules extracted from a SUBSET of files in one repository.
Summarize them into 10-25 concise capability statements describing what this
part of the codebase does. Keep them specific; preserve constraints.

Respond ONLY with a JSON array of strings. No markdown, no explanations.

Business rules:
{rules_block}
"""


def _rules_to_blocks(business_rules: dict[str, list[str]], token_budget: int):
    """
    Splits { file_path: [rules] } into text blocks, each of which encodes to at
    most `token_budget` tokens. A single file whose rules alone exceed the budget
    is further split across its individual rules so no block ever goes over.
    """
    enc = _get_encoder()
    blocks, current, size = [], [], 0

    def flush():
        nonlocal current, size
        if current:
            blocks.append("\n".join(current))
            current, size = [], 0

    for rel, rules in business_rules.items():
        # Build per-file lines, but guard against a single file exceeding budget
        lines = [f"\n{rel}:"] + [f"  - {r}" for r in rules]
        for line in lines:
            t = count_tokens(line, enc)
            if t > token_budget:
                # A single line bigger than the whole budget: hard-truncate it
                # to stay under the ceiling rather than emit an oversized request.
                if enc is not None:
                    ids = enc.encode(line)[:token_budget]
                    line = enc.decode(ids)
                    t = token_budget
                else:
                    line = line[: token_budget * 4]
                    t = token_budget
            if size + t > token_budget and current:
                flush()
            current.append(line)
            size += t
    flush()
    return blocks


def _assert_under_ceiling(prompt: str, label: str):
    """Final safety check: never send a synthesis prompt over the token ceiling."""
    n = count_tokens(prompt)
    if n > SYNTH_MAX_TOKENS:
        raise ValueError(
            f"{label} prompt is {n} tokens, over the {SYNTH_MAX_TOKENS} ceiling. "
            f"Lower SYNTH_INPUT_BUDGET."
        )
    return n


def _clean_points(points, max_points):
    """Keep only non-empty strings, de-dupe, and hard-cap the count."""
    seen, out = set(), []
    for p in points:
        if not isinstance(p, str):
            continue
        p = p.strip()
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        out.append(p)
        if len(out) >= max_points:
            break
    return out


def _text_to_blocks(text_lines, token_budget):
    """Packs pre-formatted lines into token-bounded blocks (used by REDUCE)."""
    enc = _get_encoder()
    blocks, current, size = [], [], 0
    for line in text_lines:
        t = count_tokens(line, enc)
        if size + t > token_budget and current:
            blocks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += t
    if current:
        blocks.append("\n".join(current))
    return blocks


def synthesize_repo_function(business_rules: dict[str, list[str]], client: OpenAI,
                             max_points: int = None) -> list[str]:
    """
    Map-reduce synthesis that respects the token ceiling and returns AT MOST
    `max_points` key points (default MAX_KEY_POINTS = 15).

    MAP:    split rules into token-bounded batches, summarize each.
    REDUCE: fold the batch summaries into the final key points. If the summaries
            themselves exceed the budget, reduce iteratively until they fit.
    """
    max_points = max_points or MAX_KEY_POINTS
    print("\n  Synthesizing repository function from all rules...")

    blocks = _rules_to_blocks(business_rules, SYNTH_INPUT_BUDGET)

    # Single batch: one call.
    if len(blocks) <= 1:
        rules_block = blocks[0] if blocks else ""
        prompt = SYNTHESIS_PROMPT.format(rules_block=rules_block, max_points=max_points)
        _assert_under_ceiling(prompt, "Synthesis")
        raw = call_openai(client, prompt, SYNTHESIS_MODEL)
        key_points = _clean_points(parse_json_response(raw), max_points)
        print(f"  ✓ Generated {len(key_points)} key points (max {max_points}).")
        return key_points

    # MAP — summarize each batch
    print(f"  Large repo: summarizing {len(blocks)} batches...")
    partials = []
    for i, block in enumerate(blocks, 1):
        print(f"    Batch {i}/{len(blocks)}...", flush=True)
        prompt = BATCH_SUMMARY_PROMPT.format(rules_block=block)
        _assert_under_ceiling(prompt, f"Batch {i}")
        raw = call_openai(client, prompt, SYNTHESIS_MODEL)
        partials.extend(p for p in parse_json_response(raw) if isinstance(p, str))

    # REDUCE — fold summaries into final key points, iterating if they don't fit
    round_num = 0
    while True:
        round_num += 1
        lines = [f"  - {p}" for p in partials]
        reduce_blocks = _text_to_blocks(lines, SYNTH_INPUT_BUDGET)

        if len(reduce_blocks) == 1:
            print(f"  Reducing {len(partials)} capabilities into "
                  f"≤{max_points} key points...")
            prompt = SYNTHESIS_PROMPT.format(rules_block=reduce_blocks[0],
                                             max_points=max_points)
            _assert_under_ceiling(prompt, "Reduce")
            raw = call_openai(client, prompt, SYNTHESIS_MODEL)
            key_points = _clean_points(parse_json_response(raw), max_points)
            print(f"  ✓ Generated {len(key_points)} key points "
                  f"(max {max_points}) from {len(blocks)} batches.")
            return key_points

        # Too many summaries to fit in one call — condense them a round further.
        print(f"  Reduce round {round_num}: {len(partials)} capabilities across "
              f"{len(reduce_blocks)} blocks — condensing...")
        next_partials = []
        for j, rb in enumerate(reduce_blocks, 1):
            prompt = BATCH_SUMMARY_PROMPT.format(rules_block=rb)
            _assert_under_ceiling(prompt, f"Reduce {round_num}.{j}")
            raw = call_openai(client, prompt, SYNTHESIS_MODEL)
            next_partials.extend(p for p in parse_json_response(raw) if isinstance(p, str))

        if len(next_partials) >= len(partials):
            # Not shrinking — cut losses and force a final reduce on a truncated set
            partials = next_partials[: max(len(next_partials) // 2, max_points)]
        else:
            partials = next_partials


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        print("Error: no repositories in DB. Run metadata.py first.")
        sys.exit(1)
    if requested:
        return requested
    if len(repos) == 1:
        print(f"  Using only repository in DB: {repos[0]}")
        return repos[0]
    print("  Multiple repositories — specify one with --repo:")
    for r in repos:
        print(f"    - {r}")
    sys.exit(1)


class RateLimiter:
    """
    Thread-safe limiter capping calls to `max_per_minute` across ALL workers.
    Uses a sliding 60-second window: before each call, a worker acquires a slot;
    if the last `max_per_minute` calls all happened within the past minute, it
    sleeps until the oldest one ages out.
    """
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._lock = threading.Lock()
        self._timestamps = []  # times of recent calls

    def acquire(self):
        if not self.max_per_minute or self.max_per_minute <= 0:
            return  # disabled
        while True:
            with self._lock:
                now = time.time()
                # Drop timestamps older than 60s
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                # Otherwise wait until the oldest call is 60s old
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.01
            time.sleep(max(sleep_for, 0.01))


def _process_one_file(file_meta, repo_root, max_lines, client, limiter):
    """
    Worker: extract rules for a single file. Returns (file_id, rules).
    Does NOT touch the DB — the caller owns DB writes so connections stay
    per-thread/serialized. Safe to run concurrently.

    file_meta["file_path"] starts with the repo name (e.g. "flask/src/x.py").
    repo_root is the repo's own directory on disk, which already ends in the
    repo name — so the disk read joins against repo_root.parent to avoid
    doubling it.
    """
    rel      = file_meta["file_path"]
    file_id  = file_meta["file_id"]
    language = file_meta["language"]
    name     = Path(rel).name

    code = read_file(repo_root.parent / rel, max_lines)
    rules = []
    if code.strip():
        file_json = {"name": name, "relative": rel,
                     "language": language, "content": code}
        prompt = FILE_RULES_PROMPT.format(
            filename=name, language=language,
            file_json=json.dumps(file_json, indent=2),
        )
        if limiter:
            limiter.acquire()          # shared throttle across workers
        raw = call_openai(client, prompt, MODEL, system=FILE_RULES_SYSTEM)
        rules = [r for r in parse_json_response(raw)
                 if isinstance(r, str) and r.strip()]
    return file_id, rules


def run(repo_name, repo_path, max_lines, only_unprocessed, synthesize,
        workers=1, rate_limit=0):
    repo_root = Path(repo_path)
    conn = get_connection()

    try:
        # Pull files from the DB (metadata.py must have run first)
        files = get_files(conn, repo_name, only_unprocessed=only_unprocessed)
        files = [f for f in files if f["language"] in CODE_LANGUAGES]

        if not files:
            print(f"  No files to process for '{repo_name}'.")
            if only_unprocessed:
                print("  (all files already have rules — drop --only-unprocessed to redo)")
            return

        client = get_client()
        total = len(files)
        workers = max(1, workers)
        limiter = RateLimiter(rate_limit) if rate_limit and rate_limit > 0 else None

        mode = f"{workers} workers" if workers > 1 else "sequential"
        thr  = f", ≤{rate_limit}/min" if limiter else ""
        print(f"\n  Extracting rules for {total} files (repo: {repo_name}) [{mode}{thr}]...\n")

        done = 0
        def progress(name):
            nonlocal done
            done += 1
            pct = int(done / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {done}/{total}  {name:<40}", end="", flush=True)

        if workers == 1:
            # Sequential path (unchanged behavior, plus optional rate limit)
            for file_meta in files:
                file_id, rules = _process_one_file(
                    file_meta, repo_root, max_lines, client, limiter)
                save_business_rules(conn, file_id, rules)
                conn.commit()  # per-file commit = safe resume
                progress(Path(file_meta["file_path"]).name)
        else:
            # Concurrent extraction; DB writes happen here in the main thread as
            # results arrive, so the psycopg connection is never shared.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process_one_file, fm, repo_root, max_lines, client, limiter): fm
                    for fm in files
                }
                for fut in as_completed(futures):
                    fm = futures[fut]
                    try:
                        file_id, rules = fut.result()
                        save_business_rules(conn, file_id, rules)
                        conn.commit()  # per-file commit = safe resume
                    except Exception as e:
                        print(f"\n    ! {Path(fm['file_path']).name}: {e}")
                    progress(Path(fm["file_path"]).name)

        print(f"\n\n  ✓ Rules extracted and saved to PostgreSQL.")

        # Optional synthesis from the FULL rule set in the DB
        if synthesize:
            all_rules = load_business_rules_from_db(conn, repo_name)
            if all_rules:
                key_points = synthesize_repo_function(all_rules, client)
                save_key_points(conn, repo_name, key_points)
                conn.commit()
                print(f"  ✓ {len(key_points)} key points saved to key_points table.")

    finally:
        conn.close()

    print("\n  Done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — extract business rules into PostgreSQL (OpenAI)"
    )
    parser.add_argument("--repo", help="Repository name as stored in the DB")
    parser.add_argument("--repo-path", required=True, help="Path to repository root on disk")
    parser.add_argument("--lines", type=int, default=MAX_FILE_LINES)
    parser.add_argument("--only-unprocessed", action="store_true",
                        help="Skip files whose rules were already extracted (resume)")
    parser.add_argument("--no-synthesize", action="store_true",
                        help="Skip the key-point synthesis step")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent extraction workers (default: 1)")
    parser.add_argument("--rate-limit", type=int, default=0,
                        help="Max OpenAI calls per minute across all workers (0 = unlimited)")
    args = parser.parse_args()

    if not Path(args.repo_path).exists():
        print(f"Error: repo path '{args.repo_path}' not found.")
        sys.exit(1)

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
    finally:
        conn.close()

    run(repo_name, args.repo_path, args.lines,
        args.only_unprocessed, not args.no_synthesize,
        workers=args.workers, rate_limit=args.rate_limit)


if __name__ == "__main__":
    main()