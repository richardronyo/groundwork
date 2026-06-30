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
from pathlib import Path

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

SYNTHESIS_PROMPT = """\
Your task is to produce a repository capability catalog.

For every meaningful behavior that appears repeatedly across the codebase:
- create one capability statement
- keep it specific
- avoid architectural marketing language
- preserve constraints and conditions

Output 30-100 capabilities if needed.

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


def synthesize_repo_function(business_rules: dict[str, list[str]], client: OpenAI) -> list[str]:
    """Takes { file_path: [rules] } → list of repo-level key points."""
    print("\n  Synthesizing repository function from all rules...")
    lines = []
    for rel, rules in business_rules.items():
        lines.append(f"\n{rel}:")
        for rule in rules:
            lines.append(f"  - {rule}")
    rules_block = "\n".join(lines)
    prompt = SYNTHESIS_PROMPT.format(rules_block=rules_block)
    raw = call_openai(client, prompt, SYNTHESIS_MODEL)
    key_points = parse_json_response(raw)
    print(f"  ✓ Generated {len(key_points)} key points.")
    return key_points


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


def run(repo_name, repo_path, max_lines, only_unprocessed, synthesize):
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
        print(f"\n  Extracting rules for {total} files (repo: {repo_name})...\n")

        for i, file_meta in enumerate(files):
            rel      = file_meta["file_path"]
            file_id  = file_meta["file_id"]
            language = file_meta["language"]
            name     = Path(rel).name

            pct = int((i + 1) / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {i+1}/{total}  {name:<40}", end="", flush=True)

            code = read_file(repo_root / rel, max_lines)
            rules = []
            if code.strip():
                file_json = {"name": name, "relative": rel,
                             "language": language, "content": code}
                prompt = FILE_RULES_PROMPT.format(
                    filename=name, language=language,
                    file_json=json.dumps(file_json, indent=2),
                )
                raw = call_openai(client, prompt, MODEL, system=FILE_RULES_SYSTEM)
                rules = [r for r in parse_json_response(raw)
                         if isinstance(r, str) and r.strip()]

            # Save rules + mark file processed (one commit per file = safe resume)
            save_business_rules(conn, file_id, rules)
            conn.commit()

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
        args.only_unprocessed, not args.no_synthesize)


if __name__ == "__main__":
    main()