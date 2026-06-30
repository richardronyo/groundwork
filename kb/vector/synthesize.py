"""
Groundwork — Stage 4: Key Point Synthesis → PostgreSQL

Reads business rules from the DB, synthesizes repo-level key points via OpenAI,
saves them to the key_points table. Independently runnable.

Usage:
    python3 synthesize.py --repo flask
    python3 synthesize.py            # if only one repo in the DB
"""

import sys
import argparse

from kb.vector.extract_business_rules import get_client, synthesize_repo_function
from kb.relationaldb.initialize_db import (
    get_connection, load_business_rules_from_db,
    save_key_points, list_repositories,
)


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        print("Error: no repositories in DB. Run metadata.py first.")
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


def main():
    parser = argparse.ArgumentParser(description="Synthesize key points from DB rules")
    parser.add_argument("--repo", help="Repository name as stored in the DB")
    args = parser.parse_args()

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
        business_rules = load_business_rules_from_db(conn, repo_name)
        if not business_rules:
            print(f"Error: no business rules for '{repo_name}'. Run extract_business_rules.py first.")
            sys.exit(1)

        print(f"Loaded rules for {len(business_rules)} files.")
        client = get_client()
        repo_function = synthesize_repo_function(business_rules, client)

        save_key_points(conn, repo_name, repo_function)
        conn.commit()
        print(f"Saved {len(repo_function)} key points to the key_points table.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()