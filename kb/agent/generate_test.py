#!/usr/bin/env python3
"""
Groundwork — single-shot test generation (no agent, no loop).

Gathers context itself (source file + any prior memory notes on this file),
builds one prompt, makes exactly one LLM call, and writes out the result.
No tool-calling, no revision, no execution. This is the fast/cheap baseline
counterpart to kb/agent/generate_test_agentic.py — useful as the "single-pass"
side of the paper's single-pass-vs-agentic ablation.

Usage:
    python3 -m kb.agent.generate_test_simple repos/flask flask/app.py --function create_app
"""

import argparse
import re
from pathlib import Path

from kb.llm.client import get_completion, LLMCallError
from kb.agent.memory import recall_for_file, remember_interaction
from kb.agent.rag import gather_rag_context, format_rag_context

MAX_READ_CHARS = 6000

SYSTEM_PROMPT = """You are a test-generation assistant for a Python codebase using pytest.
You will be given a source file's contents and asked to write a pytest test for it.
Cover the normal case plus at least one edge case. Reply with ONLY a single
```python fenced code block containing the complete test file — no other prose."""


def _read_source(repo_root: Path, target_file: str) -> str:
    full_path = _resolve_target_path(repo_root, target_file)
    return full_path.read_text(errors="replace")[:MAX_READ_CHARS]


def _resolve_target_path(repo_root: Path, target_file: str) -> Path:
    """target_file is meant to be relative to repo_root, but it's an easy
    mistake to paste a path copied from `tree` output that already includes
    the repo name (e.g. repo_root='repos/flask', target_file=
    'repos/flask/src/flask/app.py') — that doubles the prefix. Try the direct
    join first, then fall back to stripping a leading '<repo_name>/' before
    giving up. Same fix as store_detailed_metrics() in ingestion_pipeline.py."""
    direct = (repo_root / target_file).resolve()
    if direct.exists():
        return direct

    prefix = f"{repo_root.name}/"
    if target_file.startswith(prefix):
        stripped = (repo_root / target_file[len(prefix):]).resolve()
        if stripped.exists():
            return stripped

    raise FileNotFoundError(
        f"'{target_file}' not found under {repo_root} "
        f"(tried {direct}"
        + (f" and {stripped}" if target_file.startswith(prefix) else "")
        + ") — target_file should be relative to the repo root, without the repo name repeated."
    )


def _extract_code(response_text: str) -> str:
    """Pulls the content out of a ```python ... ``` fence; falls back to the
    raw response if the model didn't fence it (some models occasionally skip
    the fence even when told to use one)."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


def generate_test_simple(repo_root: Path, repo_name: str, target_file: str,
                          function_name: str = None, model: str = "gpt-4o",
                          provider: str = "openai", use_rag: bool = False,
                          top_n_rules: int = 5, top_n_deps: int = 3) -> str:
    source = _read_source(repo_root, target_file)

    # Gather context directly — no tool call, just a function call — from
    # whatever's already been recorded about this file in past runs.
    prior_notes = recall_for_file(repo_name, target_file)

    task = f"Source file `{target_file}`"
    if function_name:
        task += f" (write a test specifically for `{function_name}`)"
    task += f":\n\n```python\n{source}\n```"

    if use_rag:
        rag_context = gather_rag_context(
            repo_root, repo_name, target_file,
            top_n_rules=top_n_rules, top_n_deps=top_n_deps,
        )
        rag_text = format_rag_context(rag_context)
        if rag_text:
            task += "\n\n" + rag_text
        else:
            print("  (--rag requested, but no business rules/key points/dependencies "
                  "found for this file yet — has extract/embeddings/deps been run?)")

    if prior_notes:
        task += "\n\nNotes from previous work on this file:\n" + "\n".join(f"- {n}" for n in prior_notes)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    try:
        response = get_completion(messages, model=model, provider=provider)
    except LLMCallError as e:
        print(f"  ✗ Generation failed: {e}")
        return None

    test_code = _extract_code(response)

    # Store the actual prompt sent and the actual response received — not a
    # paraphrased summary — so a later run (or the paper's data collection)
    # can see exactly what context led to exactly what output.
    remember_interaction(repo_name, target_file, input_text=task, output_text=response)

    return test_code


def main():
    parser = argparse.ArgumentParser(description="Single-shot pytest test generation (no agent)")
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("target_file", help="File to test, relative to the repo root")
    parser.add_argument("--function", default=None, help="Specific function/method to target")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--rag", action="store_true",
                        help="Include retrieved context: top business rules, top key point, "
                             "and dependency file contents (requires the extract/embeddings/deps "
                             "stages to have been run for this repo)")
    parser.add_argument("--top-n-rules", type=int, default=5,
                        help="Max business rules to include with --rag (default: 5)")
    parser.add_argument("--top-n-deps", type=int, default=3,
                        help="Max dependency files to include with --rag (default: 3)")
    parser.add_argument("--out", default=None,
                        help="Where to write the test file (default: ./test_agent_simple_<name>.py, "
                             "in the current directory)")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    repo_name = repo_root.name

    test_code = generate_test_simple(
        repo_root, repo_name, args.target_file,
        function_name=args.function, model=args.model, provider=args.provider,
        use_rag=args.rag, top_n_rules=args.top_n_rules, top_n_deps=args.top_n_deps,
    )
    if test_code is None:
        return

    out_path = Path(args.out) if args.out else Path.cwd() / f"test_agent_simple_{Path(args.target_file).stem}.py"
    out_path.write_text(test_code)
    print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()