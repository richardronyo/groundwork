"""
Groundwork — Stage 3: Business Rules → PostgreSQL (Any LLM Provider)

Reads the file list from PostgreSQL, extracts business rules via a
provider-agnostic LLM client, and writes them to the business_rules table.

Usage:
    python3 extract_business_rules.py --repo flask --repo-path ./flask --provider openai --model gpt-4o
    python3 extract_business_rules.py --repo flask --repo-path ./flask --provider anthropic --model claude-3-opus
    python3 extract_business_rules.py --repo flask --repo-path ./flask --only-unprocessed
    python3 extract_business_rules.py --repo flask --repo-path ./flask --no-synthesize
"""

import json
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from kb.relationaldb.initialize_db import (
    get_connection, save_business_rules, save_key_points,
    get_files, load_business_rules_from_db, list_repositories,
)
from kb.llm.client import get_completion, LLMCallError

load_dotenv()

MAX_FILE_LINES  = 80
RETRY_DELAY     = 2

# This set must match exactly the language strings returned by tree_to_json.
# If you see mismatches, update this set.
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


def call_llm(messages, model, provider, temperature=0.2, retries=3):
    """Wrapper around get_completion with retries."""
    for attempt in range(retries):
        try:
            return get_completion(messages, model, provider, temperature=temperature)
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error: {e} — retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def parse_json_response(raw: str, context: str = "") -> list:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        print(f"\n  ⚠️  Response for {context or '(unknown)'} was valid JSON but not a list "
              f"— treating as 0 rules. Raw response started with: {raw[:120]!r}")
        return []
    except json.JSONDecodeError:
        print(f"\n  ⚠️  Could not parse JSON for {context or '(unknown)'} "
              f"— treating as 0 rules. Raw response started with: {raw[:120]!r}")
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


# ─── Synthesis helpers (unchanged logic, but using call_llm) ────────────────

SYNTH_MAX_TOKENS     = 500_000
SYNTH_OUTPUT_BUDGET  = 32_000
SYNTH_SCAFFOLD_BUDGET = 2_000
SYNTH_SAFETY_TOKENS  = 200_000
SYNTH_INPUT_BUDGET   = (SYNTH_MAX_TOKENS - SYNTH_OUTPUT_BUDGET
                        - SYNTH_SCAFFOLD_BUDGET - SYNTH_SAFETY_TOKENS)

_ENCODER_CACHE = "unset"

def _get_encoder():
    global _ENCODER_CACHE
    if _ENCODER_CACHE != "unset":
        return _ENCODER_CACHE
    try:
        import tiktoken
        try:
            _ENCODER_CACHE = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            _ENCODER_CACHE = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER_CACHE = None
    return _ENCODER_CACHE

def count_tokens(text: str, encoder=None) -> int:
    enc = encoder or _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return (len(text) + 2) // 3

def _rules_to_blocks(business_rules: dict[str, list[str]], token_budget: int):
    enc = _get_encoder()
    blocks, current, size = [], [], 0
    def flush():
        nonlocal current, size
        if current:
            blocks.append("\n".join(current))
            current, size = [], 0
    for rel, rules in business_rules.items():
        lines = [f"\n{rel}:"] + [f"  - {r}" for r in rules]
        for line in lines:
            t = count_tokens(line, enc)
            if t > token_budget:
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

def _clean_points(points, max_points):
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

def synthesize_repo_function(business_rules: dict[str, list[str]],
                             provider: str, model: str,
                             max_points: int = None) -> list[str]:
    max_points = max_points or MAX_KEY_POINTS
    print("\n  Synthesizing repository function from all rules...")

    blocks = _rules_to_blocks(business_rules, SYNTH_INPUT_BUDGET)

    if len(blocks) <= 1:
        rules_block = blocks[0] if blocks else ""
        prompt = SYNTHESIS_PROMPT.format(rules_block=rules_block, max_points=max_points)
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model, provider=provider, temperature=0.3
        )
        key_points = _clean_points(parse_json_response(raw, context="synthesis"), max_points)
        print(f"  ✓ Generated {len(key_points)} key points (max {max_points}).")
        return key_points

    print(f"  Large repo: summarizing {len(blocks)} batches...")
    partials = []
    for i, block in enumerate(blocks, 1):
        print(f"    Batch {i}/{len(blocks)}...", flush=True)
        prompt = f"""Below are business rules extracted from a SUBSET of files.
Summarize them into 10-25 concise capability statements.
Respond ONLY with a JSON array of strings.

Business rules:
{block}"""
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model, provider=provider, temperature=0.3
        )
        partials.extend(p for p in parse_json_response(raw, context=f"synthesis batch {i}/{len(blocks)}") if isinstance(p, str))

    round_num = 0
    while True:
        round_num += 1
        lines = [f"  - {p}" for p in partials]
        reduce_blocks = _text_to_blocks(lines, SYNTH_INPUT_BUDGET)

        if len(reduce_blocks) == 1:
            print(f"  Reducing {len(partials)} capabilities into ≤{max_points} key points...")
            prompt = SYNTHESIS_PROMPT.format(rules_block=reduce_blocks[0], max_points=max_points)
            raw = call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=model, provider=provider, temperature=0.3
            )
            key_points = _clean_points(parse_json_response(raw, context="synthesis reduce"), max_points)
            print(f"  ✓ Generated {len(key_points)} key points from {len(blocks)} batches.")
            return key_points

        print(f"  Reduce round {round_num}: {len(partials)} capabilities across "
              f"{len(reduce_blocks)} blocks — condensing...")
        next_partials = []
        for j, rb in enumerate(reduce_blocks, 1):
            prompt = f"""Condense these capability statements into a more concise set.
Respond ONLY with a JSON array of strings.

Capabilities:
{rb}"""
            raw = call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=model, provider=provider, temperature=0.3
            )
            next_partials.extend(p for p in parse_json_response(raw, context=f"synthesis reduce round {round_num} block {j}/{len(reduce_blocks)}") if isinstance(p, str))

        if len(next_partials) >= len(partials):
            partials = next_partials[: max(len(next_partials) // 2, max_points)]
        else:
            partials = next_partials


# ─── Main extraction logic ────────────────────────────────────────────────────

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
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._lock = threading.Lock()
        self._timestamps = []

    def acquire(self):
        if not self.max_per_minute or self.max_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = 60.0 - (now - self._timestamps[0]) + 0.01
            time.sleep(max(sleep_for, 0.01))


def _process_one_file(file_meta, repo_root, max_lines, provider, model, limiter):
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
            limiter.acquire()
        raw = call_llm(
            messages=[
                {"role": "system", "content": FILE_RULES_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            model=model, provider=provider, temperature=0.2
        )
        rules = [r for r in parse_json_response(raw, context=rel) if isinstance(r, str) and r.strip()]
    return file_id, rules


def run_extraction(repo_name: str, repo_path: str,
                   max_lines: int = 80,
                   only_unprocessed: bool = False,
                   workers: int = 1,
                   rate_limit: int = 0,
                   provider: str = "openai",
                   model: str = "gpt-4o",
                   synthesize: bool = True):
    """
    Reusable entry point for the pipeline (does not parse args).
    Now with detailed debug logging.
    """
    repo_root = Path(repo_path)
    conn = get_connection()

    try:
        # ---- DEBUG ----
        print(f"\n  DEBUG: Fetching files for repo '{repo_name}'...")
        raw_files = get_files(conn, repo_name, only_unprocessed=only_unprocessed)
        print(f"  DEBUG: Retrieved {len(raw_files)} files from the DB.")

        if raw_files:
            # Show a sample of languages present
            sample_langs = list({f["language"] for f in raw_files[:20]})
            print(f"  DEBUG: Sample languages detected: {sample_langs}")
            # Show the first file as example
            print(f"  DEBUG: Example file entry: {raw_files[0]}")
        else:
            print("  DEBUG: No files were returned. Check that the 'parse' stage has been run and that the repo name is correct.")
            print("  DEBUG: To run parse, use: python3 ingest_repo.py /path/to/repo --only parse")
            return

        # Filter by language
        files = [f for f in raw_files if f["language"] in CODE_LANGUAGES]
        print(f"  DEBUG: {len(files)} files remain after language filter (out of {len(raw_files)}).")
        if len(files) < len(raw_files):
            dropped = len(raw_files) - len(files)
            print(f"  DEBUG: {dropped} files were dropped because their language is not in CODE_LANGUAGES.")
            print(f"  DEBUG: CODE_LANGUAGES currently contains: {sorted(CODE_LANGUAGES)}")

        if not files:
            print(f"\n  No files to process for '{repo_name}'.")
            if only_unprocessed:
                print("  (all files already have rules — drop --only-unprocessed to redo)")
            else:
                print("  Possible reasons:")
                print("    1. The 'parse' stage has not been run for this repo.")
                print("       → Run: python3 ingest_repo.py /path/to/repo --only parse")
                print("    2. The language field in the DB does not match CODE_LANGUAGES.")
                print("       → Check the DB with: SELECT DISTINCT language FROM files WHERE repo_name = 'your_repo';")
                print("       → Then update CODE_LANGUAGES accordingly.")
                print("    3. The repo name is wrong or there are no code files.")
            return

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

        failed = []
        if workers == 1:
            for file_meta in files:
                try:
                    file_id, rules = _process_one_file(
                        file_meta, repo_root, max_lines, provider, model, limiter)
                    save_business_rules(conn, file_id, rules)
                    conn.commit()
                except Exception as e:
                    print(f"\n    ! {Path(file_meta['file_path']).name}: {e}")
                    failed.append(file_meta["file_path"])
                progress(Path(file_meta["file_path"]).name)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process_one_file, fm, repo_root, max_lines, provider, model, limiter): fm
                    for fm in files
                }
                for fut in as_completed(futures):
                    fm = futures[fut]
                    try:
                        file_id, rules = fut.result()
                        save_business_rules(conn, file_id, rules)
                        conn.commit()
                    except Exception as e:
                        print(f"\n    ! {Path(fm['file_path']).name}: {e}")
                        failed.append(fm["file_path"])
                    progress(Path(fm["file_path"]).name)

        print(f"\n\n  ✓ Rules extracted and saved to PostgreSQL "
              f"({total - len(failed)}/{total} files succeeded).")
        if failed:
            print(f"  ⚠️  {len(failed)} file(s) failed and were skipped:")
            for f in failed:
                print(f"      - {f}")

        if synthesize:
            all_rules = load_business_rules_from_db(conn, repo_name)
            if all_rules:
                key_points = synthesize_repo_function(all_rules, provider, model)
                save_key_points(conn, repo_name, key_points)
                conn.commit()
                print(f"  ✓ {len(key_points)} key points saved to key_points table.")
            else:
                print("  No rules found in DB to synthesize.")

    finally:
        conn.close()

    print("\n  Done.\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — extract business rules into PostgreSQL (any LLM provider)"
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
                        help="Max LLM calls per minute across all workers (0 = unlimited)")
    parser.add_argument("--provider", default="openai",
                        help="LLM provider (openai, anthropic, cohere, etc.)")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model name (e.g., gpt-4o, claude-3-opus-20240229)")
    args = parser.parse_args()

    if not Path(args.repo_path).exists():
        print(f"Error: repo path '{args.repo_path}' not found.")
        sys.exit(1)

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
    finally:
        conn.close()

    run_extraction(
        repo_name=repo_name,
        repo_path=args.repo_path,
        max_lines=args.lines,
        only_unprocessed=args.only_unprocessed,
        workers=args.workers,
        rate_limit=args.rate_limit,
        provider=args.provider,
        model=args.model,
        synthesize=not args.no_synthesize
    )


if __name__ == "__main__":
    main()