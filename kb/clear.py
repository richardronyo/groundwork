#!/usr/bin/env python3
"""
Groundwork — Knowledge Base Teardown

Clears the knowledge base across all three stores:
  - PostgreSQL  (files, business_rules, key_points)
  - Kùzu        (File / Directory nodes + their edges, embedded)
  - ChromaDB    (per-repo vector collections)

Modes:
  Full wipe (default) — removes ALL repositories from all three stores.
  Single repo (--repo) — removes just one repository from all three stores.

Usage:
    python3 -m kb.clear                    # wipe everything (asks to confirm)
    python3 -m kb.clear --repo flask       # wipe only the 'flask' repo
    python3 -m kb.clear --yes              # skip the confirmation prompt
    python3 -m kb.clear --list             # list repos, then exit
    python3 -m kb.clear --postgres-only    # limit to one store (also --kuzu-only, --chroma-only)

Requirements:
    pip install psycopg kuzu chromadb python-dotenv
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection, list_repositories,
    clear_repo, clear_all,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH    = "./chroma_db"
CHROMA_COLLECTION = "groundwork"


def collection_name_for(repo_name: str) -> str:
    """Per-repo ChromaDB collection name (matches embeddings.py)."""
    safe = "".join(c if c.isalnum() else "_" for c in (repo_name or "").lower())
    return f"{CHROMA_COLLECTION}_{safe}"


# ── PostgreSQL ────────────────────────────────────────────────────────────────

def clear_postgres(repo_name=None):
    conn = get_connection()
    try:
        if repo_name:
            removed = clear_repo(conn, repo_name)
            print(f"  PostgreSQL: removed repo '{repo_name}' "
                  f"({removed['files']} files, {removed['business_rules']} rules, "
                  f"{removed['key_points']} key points)")
        else:
            clear_all(conn)
            print("  PostgreSQL: all tables truncated (files, business_rules, key_points)")
    finally:
        conn.close()


# ── Kùzu (graph) ──────────────────────────────────────────────────────────────

def clear_kuzu(repo_name=None):
    try:
        from kb.graph.kuzu_store import (
            get_connection as kuzu_conn, clear_repo as kz_clear_repo,
            clear_all as kz_clear_all, KUZU_DB_PATH,
        )
    except ImportError:
        print("  Kùzu: kuzu not installed — skipping.")
        return

    if not Path(KUZU_DB_PATH).exists():
        print(f"  Kùzu: no {KUZU_DB_PATH} directory — nothing to clear.")
        return

    try:
        conn = kuzu_conn()
        if repo_name:
            n = kz_clear_repo(conn, repo_name)
            print(f"  Kùzu: removed {n} nodes for repo '{repo_name}'")
        else:
            n = kz_clear_all(conn)
            print(f"  Kùzu: removed {n} File/Directory nodes (all repos)")
    except Exception as e:
        print(f"  Kùzu: skipped ({e})")


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def clear_chroma(repo_name=None):
    try:
        import chromadb
    except ImportError:
        print("  ChromaDB: chromadb not installed — skipping.")
        return

    db_path = Path(CHROMA_DB_PATH)
    if not db_path.exists():
        print("  ChromaDB: no ./chroma_db directory — nothing to clear.")
        return

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if repo_name:
        # Delete just this repo's collection (try per-repo name, then legacy)
        deleted = []
        for name in (collection_name_for(repo_name), CHROMA_COLLECTION):
            try:
                client.delete_collection(name)
                deleted.append(name)
            except Exception:
                pass
        if deleted:
            print(f"  ChromaDB: deleted collection(s) {deleted}")
        else:
            print(f"  ChromaDB: no collection found for repo '{repo_name}'")
    else:
        # Full wipe: delete every collection, then remove the directory
        try:
            names = [c.name for c in client.list_collections()]
        except Exception:
            names = []
        for name in names:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        # Release the client before removing files
        del client
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        print(f"  ChromaDB: deleted {len(names)} collection(s) and removed {CHROMA_DB_PATH}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def confirm(prompt: str) -> bool:
    reply = input(prompt).strip().lower()
    return reply in ("y", "yes")


def main():
    parser = argparse.ArgumentParser(
        description="Clear the Groundwork knowledge base (PostgreSQL + Kùzu + ChromaDB)"
    )
    parser.add_argument("--repo", default=None,
                        help="Clear only this repository (default: clear everything)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt")
    parser.add_argument("--list", action="store_true",
                        help="List repositories, then exit")
    parser.add_argument("--postgres-only", action="store_true")
    parser.add_argument("--kuzu-only", action="store_true")
    parser.add_argument("--chroma-only", action="store_true")
    args = parser.parse_args()

    # --list: show what's there and exit
    if args.list:
        conn = get_connection()
        try:
            repos = list_repositories(conn)
        finally:
            conn.close()
        if repos:
            print("\n  Repositories in the knowledge base:")
            for r in repos:
                print(f"    - {r}")
            print()
        else:
            print("\n  Knowledge base is empty.\n")
        return

    # Which stores?
    only_flags = [args.postgres_only, args.kuzu_only, args.chroma_only]
    do_all_stores = not any(only_flags)
    do_pg     = do_all_stores or args.postgres_only
    do_kuzu   = do_all_stores or args.kuzu_only
    do_chroma = do_all_stores or args.chroma_only

    stores = [n for n, on in
              [("PostgreSQL", do_pg), ("Kùzu", do_kuzu), ("ChromaDB", do_chroma)] if on]

    scope = f"repository '{args.repo}'" if args.repo else "ALL repositories"

    print(f"\n  About to clear {scope} from: {', '.join(stores)}")

    if not args.yes:
        print("  This cannot be undone.")
        if not confirm("  Type 'yes' to proceed: "):
            print("  Aborted.\n")
            return

    print()
    if do_pg:
        clear_postgres(args.repo)
    if do_kuzu:
        clear_kuzu(args.repo)
    if do_chroma:
        clear_chroma(args.repo)

    print(f"\n  ✓ Done. Cleared {scope}.\n")


if __name__ == "__main__":
    main()