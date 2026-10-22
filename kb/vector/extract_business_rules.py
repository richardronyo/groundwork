"""
Groundwork — Stage 3: Business Rules → PostgreSQL (LangGraph)

Two graphs:

  File graph (one file):
      extract ──► [refine_rounds == 0] ──► END
         │
         └──► interrogate ──► approved / out of rounds ──► END
                   ▲                │
                   └──── refine ◄───┘

    `interrogate` checks the rules against the code and lists unsupported,
    missing and vague rules; `refine` rewrites the rules from that critique.
    With refine_rounds=0 this is exactly the old single-call extraction, so
    the interrogation loop is an experiment switch, not a behaviour change.

  Extraction graph (one repo):
      load_files ──Send(file)──► extract_file (×N, parallel) ──► collect ──► synthesize? ──► END

    Send fan-out replaces the ThreadPoolExecutor; concurrency is capped by the
    run config's max_concurrency (the old --workers), and the rate limit is a
    LangChain InMemoryRateLimiter shared by every call (the old --rate-limit).

Usage:
    python3 -m kb.vector.extract_business_rules --repo flask --repo-path ./flask
    python3 -m kb.vector.extract_business_rules --repo flask --repo-path ./flask --refine-rounds 2
    python3 -m kb.vector.extract_business_rules --repo flask --repo-path ./flask --provider anthropic --model claude-sonnet-4-5
    python3 -m kb.vector.extract_business_rules --repo flask --repo-path ./flask --only-unprocessed --workers 8 --rate-limit 120
"""

from __future__ import annotations

import json
import sys
import argparse
import operator
import threading
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from kb.relationaldb.initialize_db import (
    get_connection, save_business_rules, save_key_points,
    get_files, load_business_rules_from_db, list_repositories,
)
from kb.llm.client import (
    get_completion, llm_from_config, run_config, invoke_structured, LLMCallError,
    DEFAULT_MODEL,
)
from kb.llm.schemas import ExtractedRules, RuleCritique, Capabilities

load_dotenv()

MAX_FILE_LINES = 80

# Must match the language strings tree_to_json produces.
CODE_LANGUAGES = {
    "Python", "JavaScript", "TypeScript",
    "JavaScript (React)", "TypeScript (React)",
    "Java", "Kotlin", "C#", "C++", "C", "C/C++ Header",
    "Go", "Rust", "Ruby", "PHP", "Swift", "Shell", "Batch", "SQL", "R",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

FILE_RULES_SYSTEM = "You extract business rules from code."

FILE_RULES_PROMPT = """You are a software analyst.
Analyze the following parsed file JSON and extract all business rules.
Focus only on domain logic, validation rules, constraints, workflows, and decision logic.
Each rule should be a user story in the format:
"As a [role], I want [action] so that [benefit]."
where the role is typically "store owner" or "customer" depending on context,
the action describes what the system should do, and the benefit explains why.
Write this as natural, readable English.

If the file has no meaningful business logic, return an empty list.

File: {filename}
Language: {language}

Parsed File JSON:
----------------------
{file_json}
"""

INTERROGATE_SYSTEM = """You audit business rules extracted from source code.
You are strict: a rule is only supported if the code shown actually implements it.
Approve only when there is nothing unsupported, nothing important missing, and
nothing too vague to write a test for."""

INTERROGATE_PROMPT = """Source file `{filename}` ({language}):

```
{code}
```

Extracted business rules:
{rules}

Check every rule against the code, then look for behaviour the rules miss."""

REFINE_PROMPT = """Source file `{filename}` ({language}):

```
{code}
```

Current business rules:
{rules}

An auditor found these problems:
Unsupported (remove or correct): {unsupported}
Missing (add a rule for each): {missing}
Vague (make testable): {vague}

Return the complete corrected list of rules, in the same user-story format.
Keep rules the auditor did not object to unchanged."""

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

Business Rules:
{rules_block}
"""

BATCH_SUMMARY_PROMPT = """Below are business rules extracted from a SUBSET of files.
Summarize them into 10-25 concise capability statements.

Business rules:
{block}"""

CONDENSE_PROMPT = """Condense these capability statements into a more concise set.

Capabilities:
{block}"""


# ── Small helpers (kept for scripts that imported them) ───────────────────────

def call_llm(messages, model, provider, temperature=0.2, retries=3):
    """Plain-text call. Retries now happen inside the LangChain client
    (max_retries), so `retries` is passed straight through."""
    return get_completion(messages, model, provider, temperature=temperature,
                          max_retries=retries)


def parse_json_response(raw: str, context: str = "") -> list:
    """Still used by anything parsing free-text JSON; the graphs below use
    structured output instead and never call this."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        print(f"\n  ⚠️  Response for {context or '(unknown)'} was valid JSON but not a list.")
        return []
    except json.JSONDecodeError:
        print(f"\n  ⚠️  Could not parse JSON for {context or '(unknown)'}.")
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


def _clean_rules(rules) -> list[str]:
    return [r.strip() for r in rules if isinstance(r, str) and r.strip()]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "(none)"


# ══════════════════════════════════════════════════════════════════════════════
#  File graph: extract → interrogate → refine loop
# ══════════════════════════════════════════════════════════════════════════════

class FileRulesState(TypedDict, total=False):
    file_path: str            # repo-prefixed, as stored in Postgres
    language: str
    code: str
    max_rounds: int           # refine rounds allowed; 0 = single-pass extraction
    rules: list[str]
    critique: dict | None
    round: int
    # One entry per extract/refine step: what the rules were and what the
    # auditor said about them — the per-file data the extraction experiment needs.
    history: Annotated[list[dict], operator.add]


def _extract(state: FileRulesState, config) -> dict:
    name = Path(state["file_path"]).name
    file_json = {"name": name, "relative": state["file_path"],
                 "language": state["language"], "content": state["code"]}
    prompt = FILE_RULES_PROMPT.format(filename=name, language=state["language"],
                                      file_json=json.dumps(file_json, indent=2))
    out = invoke_structured(llm_from_config(config, temperature=0.2), ExtractedRules,
                            [("system", FILE_RULES_SYSTEM), ("user", prompt)])
    rules = _clean_rules(out.rules)
    return {"rules": rules, "round": 0, "critique": None,
            "history": [{"round": 0, "step": "extract", "rules": rules}]}


def _interrogate(state: FileRulesState, config) -> dict:
    prompt = INTERROGATE_PROMPT.format(
        filename=Path(state["file_path"]).name, language=state["language"],
        code=state["code"], rules=_bullets(state["rules"]))
    critique = invoke_structured(llm_from_config(config, temperature=0.0), RuleCritique,
                                 [("system", INTERROGATE_SYSTEM), ("user", prompt)])
    c = critique.model_dump()
    return {"critique": c,
            "history": [{"round": state.get("round", 0), "step": "interrogate", "critique": c}]}


def _refine(state: FileRulesState, config) -> dict:
    c = state["critique"]
    prompt = REFINE_PROMPT.format(
        filename=Path(state["file_path"]).name, language=state["language"],
        code=state["code"], rules=_bullets(state["rules"]),
        unsupported=_bullets(c["unsupported"]), missing=_bullets(c["missing"]),
        vague=_bullets(c["vague"]))
    out = invoke_structured(llm_from_config(config, temperature=0.2), ExtractedRules,
                            [("system", FILE_RULES_SYSTEM), ("user", prompt)])
    rnd = state.get("round", 0) + 1
    rules = _clean_rules(out.rules)
    return {"rules": rules, "round": rnd,
            "history": [{"round": rnd, "step": "refine", "rules": rules}]}


def _after_extract(state: FileRulesState) -> str:
    return "interrogate" if state.get("max_rounds", 0) > 0 and state["rules"] is not None else END


def _after_interrogate(state: FileRulesState) -> str:
    c = state["critique"]
    nothing_to_fix = not (c["unsupported"] or c["missing"] or c["vague"])
    if c["approved"] or nothing_to_fix or state.get("round", 0) >= state.get("max_rounds", 0):
        return END
    return "refine"


def build_file_rules_graph():
    g = StateGraph(FileRulesState)
    g.add_node("extract", _extract)
    g.add_node("interrogate", _interrogate)
    g.add_node("refine", _refine)
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", _after_extract, ["interrogate", END])
    g.add_conditional_edges("interrogate", _after_interrogate, ["refine", END])
    g.add_edge("refine", "interrogate")
    return g.compile()


_FILE_GRAPH = None


def file_rules_graph():
    global _FILE_GRAPH
    if _FILE_GRAPH is None:
        _FILE_GRAPH = build_file_rules_graph()
    return _FILE_GRAPH


def extract_rules_for_code(file_path: str, language: str, code: str,
                           refine_rounds: int = 0, config: dict | None = None) -> dict:
    """Runs the file graph on code you already have (used by experiments and
    by the translation analyzer). Returns {"rules", "history", "round"}."""
    if not code.strip():
        return {"rules": [], "history": [], "round": 0}
    out = file_rules_graph().invoke(
        {"file_path": file_path, "language": language, "code": code,
         "max_rounds": refine_rounds, "history": []},
        config or run_config())
    return {"rules": out.get("rules", []), "history": out.get("history", []),
            "round": out.get("round", 0)}


# ══════════════════════════════════════════════════════════════════════════════
#  Key point synthesis (map-reduce, unchanged logic, structured output)
# ══════════════════════════════════════════════════════════════════════════════

SYNTH_MAX_TOKENS      = 500_000
SYNTH_OUTPUT_BUDGET   = 32_000
SYNTH_SCAFFOLD_BUDGET = 2_000
SYNTH_SAFETY_TOKENS   = 200_000
SYNTH_INPUT_BUDGET    = (SYNTH_MAX_TOKENS - SYNTH_OUTPUT_BUDGET
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
        for line in [f"\n{rel}:"] + [f"  - {r}" for r in rules]:
            t = count_tokens(line, enc)
            if t > token_budget:
                if enc is not None:
                    line = enc.decode(enc.encode(line)[:token_budget])
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


def _ask_points(llm, prompt: str) -> list[str]:
    return list(invoke_structured(llm, Capabilities, [("user", prompt)]).points)


def synthesize_repo_function(business_rules: dict[str, list[str]],
                             provider: str | None = None, model: str = DEFAULT_MODEL,
                             max_points: int = None, llm=None) -> list[str]:
    """Same signature as before plus an optional ready `llm`."""
    max_points = max_points or MAX_KEY_POINTS
    llm = llm or llm_from_config(run_config(model, provider), temperature=0.3)
    print("\n  Synthesizing repository function from all rules...")

    blocks = _rules_to_blocks(business_rules, SYNTH_INPUT_BUDGET)
    if len(blocks) <= 1:
        prompt = SYNTHESIS_PROMPT.format(rules_block=blocks[0] if blocks else "",
                                         max_points=max_points)
        key_points = _clean_points(_ask_points(llm, prompt), max_points)
        print(f"  ✓ Generated {len(key_points)} key points (max {max_points}).")
        return key_points

    print(f"  Large repo: summarizing {len(blocks)} batches...")
    partials = []
    for i, block in enumerate(blocks, 1):
        print(f"    Batch {i}/{len(blocks)}...", flush=True)
        partials.extend(_ask_points(llm, BATCH_SUMMARY_PROMPT.format(block=block)))

    round_num = 0
    while True:
        round_num += 1
        reduce_blocks = _text_to_blocks([f"  - {p}" for p in partials], SYNTH_INPUT_BUDGET)
        if len(reduce_blocks) == 1:
            print(f"  Reducing {len(partials)} capabilities into ≤{max_points} key points...")
            prompt = SYNTHESIS_PROMPT.format(rules_block=reduce_blocks[0], max_points=max_points)
            key_points = _clean_points(_ask_points(llm, prompt), max_points)
            print(f"  ✓ Generated {len(key_points)} key points from {len(blocks)} batches.")
            return key_points

        print(f"  Reduce round {round_num}: {len(partials)} capabilities across "
              f"{len(reduce_blocks)} blocks — condensing...")
        next_partials = []
        for rb in reduce_blocks:
            next_partials.extend(_ask_points(llm, CONDENSE_PROMPT.format(block=rb)))
        if len(next_partials) >= len(partials):
            partials = next_partials[: max(len(next_partials) // 2, max_points)]
        else:
            partials = next_partials


# ══════════════════════════════════════════════════════════════════════════════
#  Extraction graph: whole repo
# ══════════════════════════════════════════════════════════════════════════════

class ExtractionState(TypedDict, total=False):
    repo_name: str
    repo_path: str
    max_lines: int
    only_unprocessed: bool
    refine_rounds: int
    synthesize: bool
    max_key_points: int
    files: list[dict]
    results: Annotated[list[dict], operator.add]   # one per file, from extract_file
    key_points: list[str]


class _Progress:
    """Thread-safe progress bar; extract_file runs in parallel threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0
        self.total = 0

    def reset(self, total):
        with self.lock:
            self.done, self.total = 0, total

    def tick(self, name):
        with self.lock:
            self.done += 1
            if self.total:
                pct = int(self.done / self.total * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {self.done}/{self.total}  {name:<40}", end="", flush=True)


_progress = _Progress()


def _load_files(state: ExtractionState) -> dict:
    conn = get_connection()
    try:
        raw = get_files(conn, state["repo_name"],
                        only_unprocessed=state.get("only_unprocessed", False))
    finally:
        conn.close()

    files = [f for f in raw if f["language"] in CODE_LANGUAGES]
    print(f"\n  {len(raw)} files in DB for '{state['repo_name']}', "
          f"{len(files)} after language filter.")
    if not raw:
        print("  No files returned — has the parse stage been run, and does the repo name match?")
        print("    python3 -m kb.pipeline.ingest_graph <repo_path> --only scan,parse")
    elif not files:
        if state.get("only_unprocessed"):
            print("  (all files already have rules — drop --only-unprocessed to redo)")
        else:
            langs = sorted({f['language'] for f in raw})
            print(f"  None of the stored languages are in CODE_LANGUAGES: {langs}")

    rounds = state.get("refine_rounds", 0)
    loop = f", interrogation loop ≤{rounds} round(s)" if rounds else ""
    if files:
        print(f"\n  Extracting rules for {len(files)} files{loop}...\n")
    _progress.reset(len(files))
    return {"files": files}


def _fan_out(state: ExtractionState):
    if not state["files"]:
        return "collect"
    shared = {"repo_path": state["repo_path"],
              "max_lines": state.get("max_lines", MAX_FILE_LINES),
              "refine_rounds": state.get("refine_rounds", 0)}
    return [Send("extract_file", {**shared, "file": f}) for f in state["files"]]


def _extract_file(payload: dict, config) -> dict:
    """One file: read → file graph → save. Never raises; failures are recorded
    so one bad file doesn't stop the repo."""
    fm = payload["file"]
    rel = fm["file_path"]
    result = {"file_path": rel, "ok": False, "n_rules": 0, "rounds": 0,
              "history": [], "error": None}
    try:
        disk_base = Path(payload["repo_path"]).resolve().parent   # rel starts with repo name
        code = read_file(disk_base / rel, payload["max_lines"])
        out = extract_rules_for_code(rel, fm["language"], code,
                                     payload["refine_rounds"], config)
        conn = get_connection()
        try:
            save_business_rules(conn, fm["file_id"], out["rules"])
            conn.commit()
        finally:
            conn.close()
        result.update(ok=True, n_rules=len(out["rules"]), rounds=out["round"],
                      history=out["history"])
    except Exception as e:
        result["error"] = str(e)
    _progress.tick(Path(rel).name)
    return {"results": [result]}


def _collect(state: ExtractionState) -> dict:
    results = state.get("results", [])
    if not results:
        return {}
    failed = [r for r in results if not r["ok"]]
    print(f"\n\n  ✓ Rules extracted and saved to PostgreSQL "
          f"({len(results) - len(failed)}/{len(results)} files succeeded, "
          f"{sum(r['n_rules'] for r in results)} rules).")
    if state.get("refine_rounds"):
        refined = [r for r in results if r["rounds"] > 0]
        print(f"  Interrogation loop refined {len(refined)} file(s).")
    if failed:
        print(f"  ⚠️  {len(failed)} file(s) failed and were skipped:")
        for r in failed:
            print(f"      - {r['file_path']}: {r['error']}")
    return {}


def _after_collect(state: ExtractionState) -> str:
    return "synthesize" if state.get("synthesize", True) else END


def _synthesize(state: ExtractionState, config) -> dict:
    conn = get_connection()
    try:
        all_rules = load_business_rules_from_db(conn, state["repo_name"])
        if not all_rules:
            print("  No rules found in DB to synthesize.")
            return {"key_points": []}
        key_points = synthesize_repo_function(
            all_rules, max_points=state.get("max_key_points", MAX_KEY_POINTS),
            llm=llm_from_config(config, temperature=0.3))
        save_key_points(conn, state["repo_name"], key_points)
        conn.commit()
        print(f"  ✓ {len(key_points)} key points saved to key_points table.")
        return {"key_points": key_points}
    finally:
        conn.close()


def build_extraction_graph(checkpointer=None):
    g = StateGraph(ExtractionState)
    g.add_node("load_files", _load_files)
    g.add_node("extract_file", _extract_file)
    g.add_node("collect", _collect)
    g.add_node("synthesize", _synthesize)
    g.add_edge(START, "load_files")
    g.add_conditional_edges("load_files", _fan_out, ["extract_file", "collect"])
    g.add_edge("extract_file", "collect")
    g.add_conditional_edges("collect", _after_collect, ["synthesize", END])
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=checkpointer)


def run_extraction(repo_name: str, repo_path: str,
                   max_lines: int = MAX_FILE_LINES,
                   only_unprocessed: bool = False,
                   workers: int = 1,
                   rate_limit: int = 0,
                   provider: str = "openai",
                   model: str = DEFAULT_MODEL,
                   synthesize: bool = True,
                   refine_rounds: int = 0,
                   config: dict | None = None) -> dict:
    """Same entry point the pipeline/TUI already call, plus refine_rounds.
    Returns the final graph state (results include per-file refinement
    history for the extraction experiment)."""
    config = config or run_config(model, provider, rate_limit=rate_limit,
                                  max_concurrency=max(1, workers))
    state = build_extraction_graph().invoke({
        "repo_name": repo_name, "repo_path": repo_path, "max_lines": max_lines,
        "only_unprocessed": only_unprocessed, "refine_rounds": refine_rounds,
        "synthesize": synthesize, "results": [],
    }, config)
    print("\n  Done.\n")
    return state


# ── CLI ───────────────────────────────────────────────────────────────────────

def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        print("Error: no repositories in DB. Run the parse stage first.")
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


def main():
    p = argparse.ArgumentParser(description="Groundwork — extract business rules (LangGraph)")
    p.add_argument("--repo", help="Repository name as stored in the DB")
    p.add_argument("--repo-path", required=True, help="Path to repository root on disk")
    p.add_argument("--lines", type=int, default=MAX_FILE_LINES)
    p.add_argument("--only-unprocessed", action="store_true")
    p.add_argument("--no-synthesize", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="Max files in flight at once")
    p.add_argument("--rate-limit", type=int, default=0, help="Max LLM calls per minute (0 = no limit)")
    p.add_argument("--refine-rounds", type=int, default=0,
                   help="Interrogation loop: max critique→refine rounds per file (0 = off)")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    if not Path(args.repo_path).exists():
        print(f"Error: repo path '{args.repo_path}' not found.")
        sys.exit(1)
    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
    finally:
        conn.close()

    run_extraction(repo_name, args.repo_path, args.lines, args.only_unprocessed,
                   args.workers, args.rate_limit, args.provider, args.model,
                   not args.no_synthesize, args.refine_rounds)


if __name__ == "__main__":
    main()
