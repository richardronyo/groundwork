#!/usr/bin/env python3
"""
Groundwork — Unit Test Generation

Generates unit tests for a target file (or a single function within it) by
combining two sources of grounding:
  1. The actual source code (read from disk)
  2. The business rules for that file (from PostgreSQL)
  3. Its dependencies (from Neo4j) — so the LLM knows the collaborators

The test framework is inferred from the file's language (pytest for Python,
Jest for JS/TS, JUnit for Java, xUnit for C#, etc.).

Run from the PROJECT ROOT (the directory containing kb/).

Usage:
    python3 -m kb.generate_tests --repo flask --file src/app.py
    python3 -m kb.generate_tests --repo flask --file src/auth.py --function login
    python3 -m kb.generate_tests --repo flask --file src/app.py --out tests/
    python3 -m kb.generate_tests --repo flask --file src/app.py --print

Requirements:
    pip install openai psycopg neo4j python-dotenv
"""

import os
import re
import sys
import argparse
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection, list_repositories,
    load_business_rules_from_db,
)

# Reuse the same retrieval helpers grab_context uses, so test generation and
# querying draw on one shared context layer.
from grab_context import (
    GroundworkRetriever,
    get_dependencies as kb_get_dependencies,
    get_business_rules as kb_get_business_rules,
    load_key_points_from_db,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = "gpt-5-mini"

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


# ── Framework inference ───────────────────────────────────────────────────────

FRAMEWORK_BY_EXT = {
    ".py":   ("pytest", "test_{stem}.py"),
    ".js":   ("Jest", "{stem}.test.js"),
    ".jsx":  ("Jest + React Testing Library", "{stem}.test.jsx"),
    ".ts":   ("Jest", "{stem}.test.ts"),
    ".tsx":  ("Jest + React Testing Library", "{stem}.test.tsx"),
    ".java": ("JUnit 5", "{Stem}Test.java"),
    ".cs":   ("xUnit", "{Stem}Tests.cs"),
    ".go":   ("Go testing", "{stem}_test.go"),
    ".rb":   ("RSpec", "{stem}_spec.rb"),
    ".rs":   ("Rust #[test]", "{stem}_test.rs"),
    ".php":  ("PHPUnit", "{Stem}Test.php"),
}


def infer_framework(file_path: str):
    ext = Path(file_path).suffix.lower()
    return FRAMEWORK_BY_EXT.get(ext, ("the language's standard test framework", "test_{stem}.txt"))


def output_filename(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    stem = Path(file_path).stem
    _, pattern = FRAMEWORK_BY_EXT.get(ext, (None, "test_{stem}.txt"))
    return pattern.format(stem=stem, Stem=stem[:1].upper() + stem[1:])


# ── Source + context readers ──────────────────────────────────────────────────

def read_source(repo_path: Path, rel_file: str) -> str:
    full = repo_path / rel_file
    if not full.is_file():
        sys.exit(f"Error: file not found on disk: {full}")
    return full.read_text(encoding="utf-8", errors="ignore")


def extract_function(source: str, func_name: str, language_ext: str) -> str:
    """
    Best-effort extraction of a single function/method body from source.
    For Python we use indentation; for brace languages we balance braces.
    Falls back to the whole file if the function can't be isolated.
    """
    if language_ext == ".py":
        lines = source.splitlines()
        out, capturing, indent = [], False, None
        pat = re.compile(rf"^(\s*)(async\s+)?def\s+{re.escape(func_name)}\b")
        for line in lines:
            m = pat.match(line)
            if m and not capturing:
                capturing = True
                indent = len(m.group(1))
                out.append(line)
                continue
            if capturing:
                if line.strip() == "":
                    out.append(line)
                    continue
                cur_indent = len(line) - len(line.lstrip())
                if cur_indent <= indent:
                    break
                out.append(line)
        return "\n".join(out) if out else source

    # Brace languages: find the signature, then balance { }
    idx = source.find(func_name)
    if idx == -1:
        return source
    brace = source.find("{", idx)
    if brace == -1:
        return source
    depth, i = 0, brace
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                # back up to the start of the signature line
                line_start = source.rfind("\n", 0, idx) + 1
                return source[line_start:i + 1]
        i += 1
    return source


# ── Related context (shared with grab_context) ────────────────────────────────

def gather_related_context(repo_name, rel_file, target_source, top_n=3):
    """
    Builds the same kind of multi-store context grab_context produces, focused
    on the target file:
      - dependency files + THEIR business rules (contracts of collaborators)
      - repo-level key points (overall system purpose)
      - semantically related files via the GroundworkRetriever (BERTScore →
        ChromaDB), seeded by the target file's own rules/source

    Returns a dict of preformatted text blocks for the prompt.
    """
    # 1. Direct dependencies of the target file, and their rules
    deps_map = kb_get_dependencies([rel_file], repo_name)
    dep_files = deps_map.get(rel_file, [])
    dep_rules = kb_get_business_rules(dep_files, repo_name) if dep_files else {}

    dep_lines = []
    for dep in dep_files:
        rules = dep_rules.get(dep, [])
        dep_lines.append(f"- {dep}")
        for r in rules[:4]:  # cap so the prompt stays focused
            dep_lines.append(f"    · {r}")
    deps_block = "\n".join(dep_lines) if dep_lines else "(none recorded)"

    # 2. Repo-level key points — what the whole system does
    key_points = load_key_points_from_db(repo_name)
    kp_block = "\n".join(f"- {kp}" for kp in key_points[:15]) if key_points else "(none recorded)"

    # 3. Semantically related files via the shared retriever.
    #    Seed the query with the target file's rules (fallback: a snippet of source).
    related_block = "(none found)"
    try:
        retriever = GroundworkRetriever(top_n=top_n, repo_name=repo_name, mode="file")
        seed = f"tests and behavior of {rel_file}"
        docs = retriever.invoke(seed)
        related = []
        for d in docs:
            rel = d.metadata.get("relative")
            if rel and rel != rel_file:  # don't echo the target back
                related.append(f"- {rel} (score {d.metadata.get('score', 0):.3f})")
        if related:
            related_block = "\n".join(related)
    except Exception as e:
        related_block = f"(retrieval unavailable: {e})"

    return {
        "deps_block": deps_block,
        "key_points_block": kp_block,
        "related_block": related_block,
        "dep_count": len(dep_files),
        "kp_count": len(key_points),
    }


# ── Prompt + LLM ──────────────────────────────────────────────────────────────

TEST_SYSTEM = """You are a senior test engineer who specializes in finding bugs.
You write thorough, idiomatic unit tests whose goal is to EXPOSE WEAKNESSES:
missing validation, unhandled inputs, boundary and overflow conditions, error
paths that aren't handled, and business rules the code fails to enforce.

You write two categories of tests:
  1. Tests that PASS against the current code (documenting correct behavior).
  2. Tests that PROBE suspected weaknesses. When you believe the current code
     would fail or behave wrongly for an input, still write the test asserting
     the CORRECT behavior, and mark it as expected-to-fail using the framework's
     mechanism (for pytest: @pytest.mark.xfail(reason="...")), with a reason that
     names the weakness. This keeps the suite green while documenting the gap.

You only reference APIs that exist in the given source — you never invent them."""

TEST_PROMPT = """Write unit tests for the target below using {framework}.

TARGET: {target_desc}
REPOSITORY: {repo}
FILE: {file}

--- SOURCE CODE ---
{source}

--- BUSINESS RULES FOR THIS FILE (each should map to at least one test) ---
{rules}

--- DEPENDENCIES + THEIR BUSINESS RULES (collaborators to mock; use their rules
    to understand the contracts this file relies on) ---
{deps}

--- REPOSITORY KEY POINTS (overall system purpose, for context) ---
{key_points}

--- RELATED FILES (retrieved from the knowledge base; may share behavior) ---
{related}

Requirements:
- Use {framework} idioms and conventions.
- Cover happy paths, then aggressively probe weaknesses: malformed/empty/None
  inputs, boundary and overflow values, wrong types, division-by-zero, unhandled
  exceptions, and any business rule the code may NOT actually enforce.
- Turn each business rule for THIS file into at least one explicit test, named so
  the rule it verifies is obvious. If you suspect the code violates a rule, assert
  the CORRECT behavior and mark the test expected-to-fail with a reason.
- For every test that probes a suspected weakness, mark it expected-to-fail using
  the framework's mechanism and give a reason naming the weakness.
- Use the dependencies' rules to mock collaborators faithfully.
- Mock external dependencies where appropriate (DB, network, filesystem).
- Include necessary imports and any fixtures/setup.
- Output ONLY the test file contents — no prose, no markdown fences."""


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        sys.exit("Error: OPENAI_API_KEY not set in .env or environment.")
    return OpenAI(api_key=OPENAI_API_KEY)


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        # drop first fence line and a trailing fence if present
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


# ── Weakness analysis ─────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a senior code reviewer performing a defensive audit.
You identify concrete weaknesses, uncovered cases, and likely bugs in the given
code, cross-checking it against its documented business rules. You are specific
and practical: every finding names a real input or scenario, and every finding
comes with an actionable fix. You do not pad the report with generic advice."""

ANALYSIS_PROMPT = """Audit the target below and produce a WEAKNESSES REPORT in Markdown.

TARGET: {target_desc}
FILE: {file}

--- SOURCE CODE ---
{source}

--- BUSINESS RULES FOR THIS FILE ---
{rules}

--- TESTS JUST GENERATED (some may be marked expected-to-fail) ---
{tests}

Produce a Markdown report with these sections:

# Weaknesses Report — {file}

## Summary
A 2-3 sentence overview of the code's overall robustness.

## Findings
A numbered list. For EACH finding provide:
- **Weakness**: what is wrong or uncovered (name the specific input/scenario)
- **Impact**: what breaks, and how bad it is (low / medium / high)
- **Evidence**: the function/line or the test that exposes it
- **Suggested fix**: a concrete change, with a short code snippet if helpful

## Uncovered cases
A bullet list of scenarios the current tests/code do NOT handle but should.

## Rule compliance
For each business rule, state whether the code appears to ENFORCE it, and if not,
what to add.

Be concrete and reference actual identifiers from the source. Output only the
Markdown report."""


def run_analysis(client, target_desc, rel_file, source, rules_block, test_code):
    """Second LLM pass: produce a Markdown weaknesses report with fixes."""
    prompt = ANALYSIS_PROMPT.format(
        target_desc=target_desc, file=rel_file,
        source=source, rules=rules_block, tests=test_code,
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return strip_fences(resp.choices[0].message.content)


# ── Main ──────────────────────────────────────────────────────────────────────

def find_top_file_for_prompt(prompt, repo_name, min_score=0.3):
    """Uses the shared GroundworkRetriever (file-level) to find the single most
    associated file for a prompt. Returns (rel_file, score) or (None, None)."""
    retriever = GroundworkRetriever(top_n=1, repo_name=repo_name,
                                    mode="file", min_score=min_score)
    docs = retriever.invoke(prompt)
    if not docs:
        return None, None
    top = docs[0]
    return top.metadata.get("relative"), top.metadata.get("score")


IMPORT_PATTERNS = {
    ".py": [
        # from module import a, b, c
        (r'^\s*from\s+([\w.]+)\s+import\s+(.+)$', "from"),
        # import module
        (r'^\s*import\s+([\w.]+)', "plain"),
    ],
}


def find_imported_functions(source, rel_file, repo_name):
    """
    Parses the target file's imports and resolves them against the repo's
    dependency files (from Neo4j) to find imported FUNCTIONS worth testing.

    Returns a list of (dep_file, function_name) pairs. Only functions imported
    from files that are actually in this repo's knowledge base are returned.
    """
    ext = Path(rel_file).suffix.lower()
    patterns = IMPORT_PATTERNS.get(ext)
    if not patterns:
        return []

    # Dependency files this file imports from (repo-scoped, from Neo4j)
    deps_map = kb_get_dependencies([rel_file], repo_name)
    dep_files = deps_map.get(rel_file, [])
    # Index dep files by module stem for resolution: "helpers" -> "flask/helpers.py"
    by_stem = {}
    for d in dep_files:
        by_stem[Path(d).stem] = d

    import re
    found = []
    for line in source.splitlines():
        for pat, kind in patterns:
            m = re.match(pat, line)
            if not m:
                continue
            if kind == "from":
                module = m.group(1)              # e.g. "flask.helpers" or "helpers"
                names = m.group(2)
                mod_stem = module.split(".")[-1]  # last segment
                dep_file = by_stem.get(mod_stem)
                if not dep_file:
                    continue
                # Split imported names: "a, b as c, d"
                for raw in names.split(","):
                    name = raw.strip().split(" as ")[0].strip().strip("()")
                    # Heuristic: function-like names (skip * and CONSTANTS/Classes optional)
                    if name and name != "*":
                        found.append((dep_file, name))
    # De-dupe
    seen, result = set(), []
    for pair in found:
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        sys.exit("Error: no repositories in the knowledge base. Ingest one first.")
    if requested:
        if requested not in repos:
            sys.exit(f"Error: repo '{requested}' not found. Available: {', '.join(repos)}")
        return requested
    if len(repos) == 1:
        return repos[0]
    sys.exit("Multiple repositories exist — specify one with --repo: " + ", ".join(repos))


def generate_for_target(client, repo_name, repo_path, rel_file, rules_by_file,
                        function=None, related=3, analyze=True,
                        to_stdout=False, out_dir="generated_tests"):
    """
    Generates tests (and optionally a weaknesses report) for one target —
    either a whole file or a single function within it. Reusable so the same
    logic runs for the top file AND for each imported function.
    """
    source = read_source(repo_path, rel_file)
    if function:
        source = extract_function(source, function, Path(rel_file).suffix.lower())
        target_desc = f"function '{function}' in {rel_file}"
    else:
        target_desc = f"file {rel_file}"

    framework, _ = infer_framework(rel_file)
    file_rules = rules_by_file.get(rel_file, [])
    rules_block = "\n".join(f"- {r}" for r in file_rules) if file_rules else "(none recorded)"
    ctx = gather_related_context(repo_name, rel_file, source, top_n=related)

    print(f"\n  ── Target: {target_desc} ──")
    print(f"     Framework   : {framework}")
    print(f"     File rules  : {len(file_rules)}")
    print(f"     Dependencies: {ctx['dep_count']}")
    print(f"     Generating tests...")

    prompt = TEST_PROMPT.format(
        framework=framework, target_desc=target_desc,
        repo=repo_name, file=rel_file,
        source=source, rules=rules_block,
        deps=ctx["deps_block"], key_points=ctx["key_points_block"],
        related=ctx["related_block"],
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": TEST_SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    test_code = strip_fences(resp.choices[0].message.content)

    report = None
    if analyze:
        print("     Analyzing for weaknesses...")
        report = run_analysis(client, target_desc, rel_file, source, rules_block, test_code)

    # Build output name
    out_name = output_filename(rel_file)
    if function:
        stem, suffix = Path(out_name).stem, Path(out_name).suffix
        out_name = f"{stem}_{function}{suffix}"

    if to_stdout:
        print("\n" + "=" * 70 + f"  TESTS — {target_desc}\n")
        print(test_code)
        if report:
            print("\n" + "=" * 70 + f"  WEAKNESSES — {target_desc}\n")
            print(report)
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / out_name).write_text(test_code, encoding="utf-8")
    print(f"     ✓ Wrote tests  → {out / out_name}")
    if report:
        report_name = Path(out_name).stem + "_weaknesses.md"
        (out / report_name).write_text(report, encoding="utf-8")
        print(f"     ✓ Wrote report → {out / report_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate unit tests from the Groundwork KB")
    parser.add_argument("--repo", default=None, help="Repository name")
    parser.add_argument("--file", default=None,
                        help="Target file (relative path within the repo)")
    parser.add_argument("--prompt", default=None,
                        help="Instead of --file: find the file most associated with this prompt")
    parser.add_argument("--function", default=None, help="Optional: a single function/method to target")
    parser.add_argument("--follow-imports", action="store_true",
                        help="Also test functions this file imports from repo dependencies")
    parser.add_argument("--repo-path", default=None,
                        help="Path to the repo on disk (default: ./repos/<repo>)")
    parser.add_argument("--out", default="generated_tests",
                        help="Directory to write the test files (default: ./generated_tests)")
    parser.add_argument("--print", dest="to_stdout", action="store_true",
                        help="Print to stdout instead of writing files")
    parser.add_argument("--related", type=int, default=3,
                        help="How many related files to retrieve for context (default: 3)")
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="Min retrieval score when using --prompt (default: 0.3)")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Skip the weaknesses report (generate tests only)")
    args = parser.parse_args()

    if not args.file and not args.prompt:
        sys.exit("Error: pass either --file <path> or --prompt \"...\"")

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
        rules_by_file = load_business_rules_from_db(conn, repo_name)
    finally:
        conn.close()

    # Resolve repo path on disk
    repo_path = Path(args.repo_path) if args.repo_path else Path("./repos") / repo_name
    if not repo_path.is_dir():
        alt = Path(repo_name)
        repo_path = alt if alt.is_dir() else repo_path
    if not repo_path.is_dir():
        sys.exit(f"Error: repo path not found. Pass --repo-path. Tried: {repo_path}")

    # Resolve the target file — either given directly or found from the prompt
    if args.prompt:
        print(f"  Finding the file most associated with: \"{args.prompt}\"")
        rel_file, score = find_top_file_for_prompt(args.prompt, repo_name, args.min_score)
        if not rel_file:
            sys.exit("  No file passed the retrieval threshold. Try a different prompt "
                     "or lower --min-score.")
        print(f"  Top file: {rel_file}  (score {score:.3f})")
    else:
        rel_file = args.file

    client = get_client()

    # 1. Generate tests for the primary target
    generate_for_target(
        client, repo_name, repo_path, rel_file, rules_by_file,
        function=args.function, related=args.related,
        analyze=not args.no_analyze, to_stdout=args.to_stdout, out_dir=args.out,
    )

    # 2. Optionally follow imports: test functions imported from repo dependencies
    if args.follow_imports and not args.function:
        source = read_source(repo_path, rel_file)
        imported = find_imported_functions(source, rel_file, repo_name)
        if imported:
            print(f"\n  Following {len(imported)} imported function(s) from dependencies...")
            for dep_file, func_name in imported:
                try:
                    generate_for_target(
                        client, repo_name, repo_path, dep_file, rules_by_file,
                        function=func_name, related=args.related,
                        analyze=not args.no_analyze, to_stdout=args.to_stdout, out_dir=args.out,
                    )
                except SystemExit as e:
                    print(f"     (skipped {func_name} in {dep_file}: {e})")
        else:
            print("\n  No imported functions resolved to repo dependency files.")

    if not args.to_stdout:
        print(f"\n    Review before running — generated tests are a starting point, not ground truth.")


if __name__ == "__main__":
    main()