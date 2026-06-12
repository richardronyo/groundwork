"""
Groundwork — Business Rules Extractor
Reads each code file, sends it to OpenAI to extract business rules,
then synthesizes all rules into key points describing the repo's function.

Outputs:
    business_rules.json  — { filename: [rule1, rule2, ...] }
    repo_function.json   — ["key point 1", "key point 2", ...]

Usage:
    python3 extract_rules.py <files.json> --repo <path/to/repo>
    python3 extract_rules.py flask_files.json --repo ./flask
    python3 extract_rules.py flask_files.json --repo ./flask --lines 100

Requirements:
    pip install openai python-dotenv
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5-mini"
SYNTHESIS_MODEL = "gpt-5"

MAX_FILE_LINES = 80
RETRY_DELAY = 2

CODE_LANGUAGES = {
    "Python", "JavaScript", "TypeScript",
    "JavaScript (React)", "TypeScript (React)",
    "Java", "Kotlin", "C#", "C++", "C", "C/C++ Header",
    "Go", "Rust", "Ruby", "PHP", "Swift", "Shell", "Batch", "SQL",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

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

Example:
[
  "As a customer, I want the system to prevent me from ordering out-of-stock items so that I don't place orders that cannot be fulfilled.",
  "As a store owner, I want all admin endpoints to require authentication so that unauthorized users cannot access sensitive data."
]

File: {filename}
Language: {language}

Parsed File JSON:
----------------------
{file_json}
"""

SYNTHESIS_PROMPT = """\
You are a senior software architect. Below are business rules extracted from \
every file in a software repository.

Your task: synthesize these into a concise list of KEY POINTS that describe \
the overall function and purpose of this repository as a whole system.

Guidelines:
- Identify the core domain and what problem this software solves
- Group related rules into higher-level themes
- Aim for 8-15 key points — enough to be comprehensive, few enough to be useful
- Each point should describe a meaningful capability or constraint of the system
- Write from the perspective of what the system DOES, not how it works internally
- Avoid duplicates and overly technical implementation details

Respond ONLY with a JSON array of strings. No preamble, no markdown, no explanation.

Business rules by file:
{rules_block}
"""

# ── OpenAI Client ─────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in .env or environment.")
        sys.exit(1)

    return OpenAI(api_key=OPENAI_API_KEY)


def call_openai(
    client: OpenAI,
    prompt: str,
    model: str,
    system: str = None,
    retries: int = 3,
) -> str:

    for attempt in range(retries):
        try:
            messages = []

            if system:
                messages.append({
                    "role": "system",
                    "content": system,
                })

            messages.append({
                "role": "user",
                "content": prompt,
            })

            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error: {e} — retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise


def parse_json_response(raw: str) -> list:
    """Safely parse LLM JSON response, stripping any accidental markdown fences."""
    text = raw.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            lines[1:-1]
            if lines[-1].strip() == "```"
            else lines[1:]
        )

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


# ── File Reader ───────────────────────────────────────────────────────────────

def read_file(path: Path, max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []

            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(
                        f"... ({max_lines} lines shown, file continues)"
                    )
                    break

                lines.append(line)

        return "".join(lines)

    except OSError:
        return ""


# ── Step 1: Extract per-file business rules ───────────────────────────────────

def extract_file_rules(
    files: list[dict],
    repo_root: Path,
    client: OpenAI,
    max_lines: int,
) -> dict[str, list[str]]:

    business_rules = {}
    total = len(files)

    print(f"\n  Extracting business rules from {total} files...\n")

    for i, file_meta in enumerate(files):
        rel = file_meta["relative"]
        name = file_meta["name"]
        language = file_meta.get("language", "")
        full_path = repo_root / rel

        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)

        print(
            f"\r  [{bar}] {i+1}/{total}  {name:<40}",
            end="",
            flush=True,
        )

        code = read_file(full_path, max_lines)

        if not code.strip():
            continue

        file_json = {
            "name": name,
            "relative": rel,
            "language": language,
            "content": code,
        }

        prompt = FILE_RULES_PROMPT.format(
            filename=name,
            language=language,
            file_json=json.dumps(file_json, indent=2),
        )

        raw = call_openai(
            client,
            prompt,
            MODEL,
            system=FILE_RULES_SYSTEM,
        )

        rules = parse_json_response(raw)

        valid_rules = [
            r for r in rules
            if isinstance(r, str) and r.strip()
        ]

        if valid_rules:
            business_rules[rel] = valid_rules

    print(f"\n\n  ✓ Extracted rules from {len(business_rules)} files.")
    return business_rules


# ── Step 2: Synthesize into repo-level key points ─────────────────────────────

def synthesize_repo_function(
    business_rules: dict[str, list[str]],
    client: OpenAI,
) -> list[str]:

    print("\n  Synthesizing repository function from all rules...")

    lines = []

    for rel, rules in business_rules.items():
        lines.append(f"\n{rel}:")

        for rule in rules:
            lines.append(f"  - {rule}")

    rules_block = "\n".join(lines)

    prompt = SYNTHESIS_PROMPT.format(
        rules_block=rules_block
    )

    raw = call_openai(
        client,
        prompt,
        SYNTHESIS_MODEL,
    )

    key_points = parse_json_response(raw)

    print(f"  ✓ Generated {len(key_points)} key points.")
    return key_points


# ── Output ────────────────────────────────────────────────────────────────────

def save_outputs(
    business_rules: dict[str, list[str]],
    key_points: list[str],
    output_dir: Path,
):
    rules_path = output_dir / "business_rules.json"
    func_path = output_dir / "repo_function.json"

    with open(rules_path, "w") as f:
        json.dump(business_rules, f, indent=2)

    with open(func_path, "w") as f:
        json.dump(key_points, f, indent=2)

    print(f"\n  Saved → {rules_path}")
    print(f"  Saved → {func_path}")


def print_summary(
    business_rules: dict,
    key_points: list,
):
    total_rules = sum(
        len(v) for v in business_rules.values()
    )

    print("\n  ┌─ Summary ───────────────────────────────────┐")
    print(f"  │  Files with rules : {len(business_rules):<6}                  │")
    print(f"  │  Total rules      : {total_rules:<6}                  │")
    print(f"  │  Key points       : {len(key_points):<6}                  │")
    print("  └─────────────────────────────────────────────┘")

    print("\n  Repository key points:")

    for i, point in enumerate(key_points, 1):
        print(f"    {i:>2}. {point}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — extract business rules and repo function via OpenAI"
    )

    parser.add_argument(
        "json_file",
        help="Path to JSON file list from tree_to_json.py",
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the repository root on disk",
    )

    parser.add_argument(
        "--lines",
        type=int,
        default=MAX_FILE_LINES,
        help=f"Max lines to read per file (default: {MAX_FILE_LINES})",
    )

    parser.add_argument(
        "--output",
        default=".",
        help="Directory to save output JSON files (default: .)",
    )

    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Only extract per-file rules, skip synthesis step",
    )

    args = parser.parse_args()

    json_path = Path(args.json_file)
    repo_root = Path(args.repo)
    output_dir = Path(args.output)

    if not json_path.exists():
        print(f"Error: '{json_path}' not found.")
        sys.exit(1)

    if not repo_root.exists():
        print(f"Error: repo '{repo_root}' not found.")
        sys.exit(1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(json_path) as f:
        all_files = json.load(f)

    files = [
        f
        for f in all_files
        if f.get("language") in CODE_LANGUAGES
    ]

    print(f"\n  Loaded {len(files)} code files from {json_path}")
    print(f"  Reading up to {args.lines} lines per file")
    print(
        f"  Using {MODEL} for file rules, "
        f"{SYNTHESIS_MODEL} for synthesis"
    )

    client = get_client()

    business_rules = extract_file_rules(
        files,
        repo_root,
        client,
        args.lines,
    )

    key_points = []

    if not args.rules_only and business_rules:
        key_points = synthesize_repo_function(
            business_rules,
            client,
        )

    save_outputs(
        business_rules,
        key_points,
        output_dir,
    )

    print_summary(
        business_rules,
        key_points,
    )


if __name__ == "__main__":
    main()