#!/usr/bin/env python3
"""
Groundwork — test generation CLI (single-pass or agentic, via LangGraph).

The graphs live in kb/agent/test_graph.py. This module keeps the old
generate_test_simple() entry point (same signature, same return value) and
adds generate_test() for the agentic mode, which replaces
kb/agent/generate_test_agentic.py.

Usage:
    python3 -m kb.agent.generate_test repos/flask src/flask/app.py --function create_app
    python3 -m kb.agent.generate_test repos/flask src/flask/app.py --rag --agentic --max-attempts 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kb.llm.client import run_config, LLMCallError, DEFAULT_MODEL
from kb.agent.test_graph import build_test_graph


def generate_test(repo_root: Path, repo_name: str, target_file: str,
                  function_name: str | None = None, mode: str = "single",
                  model: str = DEFAULT_MODEL, provider: str = "openai",
                  use_rag: bool = False, use_memory: bool = True, use_bsg: bool = True,
                  top_n_rules: int = 5, top_n_deps: int = 3, max_attempts: int = 3,
                  config: dict | None = None) -> dict:
    """Runs one test-generation graph and returns its final state
    (test_code, run, attempts, prompt, ...)."""
    graph = build_test_graph(mode)
    return graph.invoke({
        "repo_root": str(Path(repo_root).resolve()), "repo_name": repo_name,
        "target_file": target_file, "function_name": function_name, "mode": mode,
        "use_rag": use_rag, "use_memory": use_memory, "use_bsg": use_bsg,
        "top_n_rules": top_n_rules, "top_n_deps": top_n_deps,
        "max_attempts": max_attempts, "attempts": [],
    }, config or run_config(model, provider))


def generate_test_simple(repo_root: Path, repo_name: str, target_file: str,
                         function_name: str = None, model: str = DEFAULT_MODEL,
                         provider: str = "openai", use_rag: bool = False,
                         top_n_rules: int = 5, top_n_deps: int = 3) -> str | None:
    """Unchanged contract: one LLM call, returns the test code (or None on failure)."""
    try:
        state = generate_test(repo_root, repo_name, target_file, function_name, "single",
                              model, provider, use_rag, top_n_rules=top_n_rules,
                              top_n_deps=top_n_deps)
    except LLMCallError as e:
        print(f"  ✗ Generation failed: {e}")
        return None
    return state["test_code"]


def main():
    p = argparse.ArgumentParser(description="pytest test generation (LangGraph)")
    p.add_argument("repo", help="Path to the repository")
    p.add_argument("target_file", help="File to test, relative to the repo root")
    p.add_argument("--function", default=None, help="Specific function/method to target")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--agentic", action="store_true",
                   help="Run tests and loop through reflect → regenerate until they pass")
    p.add_argument("--max-attempts", type=int, default=3, help="Agentic mode: generate attempts")
    p.add_argument("--rag", action="store_true",
                   help="Include business rules, top key point and dependency sources")
    p.add_argument("--no-memory", action="store_true", help="Don't read or write agent memory")
    p.add_argument("--no-bsg", action="store_true", help="Don't include the file's stored BSG")
    p.add_argument("--top-n-rules", type=int, default=5)
    p.add_argument("--top-n-deps", type=int, default=3)
    p.add_argument("--out", default=None,
                   help="Where to write the test (default: ./test_agent_<mode>_<name>.py)")
    p.add_argument("--report", default=None, help="Also write the attempt history as JSON")
    args = p.parse_args()

    repo_root = Path(args.repo).resolve()
    mode = "agentic" if args.agentic else "single"
    try:
        state = generate_test(repo_root, repo_root.name, args.target_file, args.function, mode,
                              args.model, args.provider, args.rag, not args.no_memory,
                              not args.no_bsg, args.top_n_rules, args.top_n_deps,
                              args.max_attempts)
    except LLMCallError as e:
        print(f"  ✗ Generation failed: {e}")
        return

    out = Path(args.out) if args.out else Path.cwd() / f"test_agent_{mode}_{Path(args.target_file).stem}.py"
    out.write_text(state["test_code"])
    print(f"  Written to {out}")
    if state.get("run") is not None:
        status = "passing" if state["run"]["passed"] else "still failing"
        print(f"  {status} after {state['attempt']} attempt(s)")
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"mode": mode, "target": args.target_file, "attempts": state.get("attempts", []),
             "reflection": state.get("reflection")}, indent=2))
        print(f"  Report written to {args.report}")


if __name__ == "__main__":
    main()
