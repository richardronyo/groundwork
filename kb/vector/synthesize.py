"""
Groundwork — Stage 4: Key Point Synthesis → PostgreSQL (Any LLM)

Reads business rules from the DB, synthesizes key points via a provider-agnostic
LLM client, and saves them to the key_points table.

Usage:
    python3 synthesize.py --repo flask --provider openai --model gpt-4o
    python3 synthesize.py --repo flask --provider anthropic --model claude-3-opus
"""

import sys
import argparse
from kb.relationaldb.initialize_db import (
    get_connection, load_business_rules_from_db, save_key_points, list_repositories
)
from kb.vector.extract_business_rules import synthesize_repo_function, MAX_KEY_POINTS


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        print("Error: no repositories in DB.")
        sys.exit(1)
    if requested:
        return requested
    if len(repos) == 1:
        print(f"Using only repository in DB: {repos[0]}")
        return repos[0]
    print("Multiple repositories — specify one with --repo:")
    for r in repos:
        print(f"  - {r}")
    sys.exit(1)


def run_synthesis(repo_name: str, provider: str, model: str, max_key_points: int = 15):
    """Reusable entry point for the pipeline."""
    conn = get_connection()
    try:
        rules = load_business_rules_from_db(conn, repo_name)
        if not rules:
            print(f"Error: no business rules for '{repo_name}'. Run extract_business_rules.py first.")
            return
        print(f"Loaded rules for {len(rules)} files.")
        key_points = synthesize_repo_function(rules, provider, model, max_key_points)
        save_key_points(conn, repo_name, key_points)
        conn.commit()
        print(f"Saved {len(key_points)} key points to the key_points table.")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Synthesize key points from DB rules")
    parser.add_argument("--repo", help="Repository name as stored in the DB")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--max-key-points", type=int, default=MAX_KEY_POINTS)
    args = parser.parse_args()

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
    finally:
        conn.close()

    run_synthesis(repo_name, args.provider, args.model, args.max_key_points)


if __name__ == "__main__":
    main()