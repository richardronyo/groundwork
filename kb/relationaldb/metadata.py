#!/usr/bin/env python3
"""
Groundwork — Stage 1b: File Metrics → PostgreSQL

Parses every file's metrics (via parser.py) and saves them to the files table.
No JSON. No business rules here — just metrics. Idempotent.

Usage:
    python3 metadata.py ./flask
    python3 metadata.py ./flask --python-only
"""

import sys
import argparse
from pathlib import Path

import kb.relationaldb.parser as repo_parser
from kb.relationaldb.initialize_db import get_connection, save_file


def run(repo_root: str, python_only: bool = False):
    repo_path = Path(repo_root)
    repo_name = repo_path.name

    extensions = (".py",) if python_only else None
    print(f"\n  Analyzing {repo_path} ...")
    metrics_map = repo_parser.analyze_repo(repo_root, extensions=extensions)
    print(f"  Found {len(metrics_map)} files.")

    conn = get_connection()
    try:
        saved = 0
        total = len(metrics_map)
        for i, (rel_path, data) in enumerate(metrics_map.items()):
            pct = int((i + 1) / total * 40) if total else 40
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {i+1}/{total}", end="", flush=True)

            save_file(conn, repo_name, rel_path, data["language"], data["metrics"])
            saved += 1
        conn.commit()
        print(f"\n\n  ✓ Saved metrics for {saved} files to PostgreSQL.\n")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Groundwork — file metrics to PostgreSQL")
    parser.add_argument("repo", help="Path to repository root")
    parser.add_argument("--python-only", action="store_true")
    args = parser.parse_args()

    if not Path(args.repo).exists():
        print(f"Error: '{args.repo}' not found.")
        sys.exit(1)

    run(args.repo, args.python_only)


if __name__ == "__main__":
    main()