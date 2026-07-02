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

TEST_SYSTEM = """You are a senior test engineer. You write thorough, correct,
idiomatic unit tests. You cover the happy path, edge cases, boundary conditions,
error handling, and any business rules provided. You only test behavior that is
supported by the given source code — you never invent APIs that don't exist."""

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
- Cover happy paths, edge cases, boundaries, and error handling.
- Turn each business rule for THIS file into at least one explicit test, and name
  the test so the rule it verifies is obvious.
- Use the dependencies' rules to mock collaborators faithfully (respect their
  documented contracts rather than inventing behavior).
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


# ── Main ──────────────────────────────────────────────────────────────────────

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


def main():
    parser = argparse.ArgumentParser(description="Generate unit tests from the Groundwork KB")
    parser.add_argument("--repo", default=None, help="Repository name")
    parser.add_argument("--file", required=True, help="Target file (relative path within the repo)")
    parser.add_argument("--function", default=None, help="Optional: a single function/method to target")
    parser.add_argument("--repo-path", default=None,
                        help="Path to the repo on disk (default: ./repos/<repo>)")
    parser.add_argument("--out", default="generated_tests",
                        help="Directory to write the test file (default: ./generated_tests)")
    parser.add_argument("--print", dest="to_stdout", action="store_true",
                        help="Print to stdout instead of writing a file")
    parser.add_argument("--related", type=int, default=3,
                        help="How many related files to retrieve for context (default: 3)")
    args = parser.parse_args()

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
        rules_by_file = load_business_rules_from_db(conn, repo_name)
    finally:
        conn.close()

    # Resolve the repo path on disk
    repo_path = Path(args.repo_path) if args.repo_path else Path("./repos") / repo_name
    if not repo_path.is_dir():
        # Fall back to a sibling dir named after the repo
        alt = Path(repo_name)
        repo_path = alt if alt.is_dir() else repo_path
    if not repo_path.is_dir():
        sys.exit(f"Error: repo path not found. Pass --repo-path. Tried: {repo_path}")

    rel_file = args.file
    source = read_source(repo_path, rel_file)

    # Narrow to a single function if requested
    if args.function:
        source = extract_function(source, args.function, Path(rel_file).suffix.lower())
        target_desc = f"function '{args.function}' in {rel_file}"
    else:
        target_desc = f"file {rel_file}"

    framework, _ = infer_framework(rel_file)
    file_rules = rules_by_file.get(rel_file, [])
    rules_block = "\n".join(f"- {r}" for r in file_rules) if file_rules else "(none recorded)"

    # Gather the richer, grab_context-style context around this file
    ctx = gather_related_context(repo_name, rel_file, source, top_n=args.related)

    print(f"  Repo       : {repo_name}")
    print(f"  Target     : {target_desc}")
    print(f"  Framework  : {framework}")
    print(f"  File rules  : {len(file_rules)}")
    print(f"  Dependencies: {ctx['dep_count']} (with their rules)")
    print(f"  Key points  : {ctx['kp_count']}")
    print(f"  Generating tests...")

    prompt = TEST_PROMPT.format(
        framework=framework, target_desc=target_desc,
        repo=repo_name, file=rel_file,
        source=source, rules=rules_block,
        deps=ctx["deps_block"],
        key_points=ctx["key_points_block"],
        related=ctx["related_block"],
    )

    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TEST_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    test_code = strip_fences(resp.choices[0].message.content)

    if args.to_stdout:
        print("\n" + "=" * 70 + "\n")
        print(test_code)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = output_filename(rel_file)
    if args.function:
        stem = Path(out_name).stem
        suffix = Path(out_name).suffix
        out_name = f"{stem}_{args.function}{suffix}"
    out_path = out_dir / out_name
    out_path.write_text(test_code, encoding="utf-8")

    print(f"\n  ✓ Wrote {out_path}")
    print(f"    Review before running — generated tests are a starting point, not ground truth.")


if __name__ == "__main__":
    main()