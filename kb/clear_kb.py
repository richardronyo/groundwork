#!/usr/bin/env python3
"""
Groundwork — clear the knowledge base.

Removes what the pipeline built, from every store, for one repository or all:

  postgres   files, business_rules, key_points, classes, class_attributes,
             functions, bsg_specs, translations, agent_memory
             (ai_providers — your stored API keys — is never touched unless
             you pass --include-keys)
  kuzu       File / Directory nodes and their CONTAINS, SAME_DIR, DEPENDS_ON edges
  chroma     groundwork_<repo> (key-point vectors) and
             groundwork_memory_<repo> (agent memory) collections
  outputs    outputs/translations/<repo>/ (translation results)
  cache      cache/vectors/ (BERTScore vector cache; only with --all,
             since it isn't keyed by repo)

Usage:
    python3 -m kb.clear_kb --repo flask --dry-run       # show what would go
    python3 -m kb.clear_kb --repo flask                  # asks before deleting
    python3 -m kb.clear_kb --repo flask --repo shop -y
    python3 -m kb.clear_kb --all                         # every repo, every store
    python3 -m kb.clear_kb --all --only kuzu,chroma      # just these stores
    python3 -m kb.clear_kb --repo flask --keep-memory    # keep agent memory
    python3 -m kb.clear_kb --all --drop                  # hard reset: drop tables,
                                                         # delete the Kùzu file and
                                                         # Chroma folder, re-create schema

Close any Jupyter kernel holding the Kùzu graph first — Kùzu allows only one
writer, so an open kernel makes the kuzu step fail with a lock error (the other
stores are still cleared).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

STORES = ["postgres", "kuzu", "chroma", "outputs", "cache"]

OUTPUTS_DIR = Path("outputs") / "translations"
CACHE_DIR = Path("cache") / "vectors"
CHROMA_PREFIX = "groundwork_"


def _safe(name: str) -> str:
    """Same sanitising embeddings.py / memory.py use for collection names."""
    return "".join(c if c.isalnum() else "_" for c in name.lower())


def _chroma_path() -> str:
    try:
        from kb.vector.embeddings import CHROMA_DB_PATH
        return CHROMA_DB_PATH
    except ImportError:
        return "./chroma_db"


def _kuzu_path() -> str:
    from kb.graph.kuzu_store import KUZU_DB_PATH
    return KUZU_DB_PATH


# ── Discovery: which repos and how much is stored ─────────────────────────────

def postgres_status(repos: list[str] | None):
    """{repo: {table: count}} — or raises if Postgres is unreachable."""
    from kb.relationaldb.initialize_db import get_connection, list_repositories
    conn = get_connection()
    try:
        names = repos if repos is not None else list_repositories(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT repository_name FROM agent_memory")
            mem_repos = [r[0] for r in cur.fetchall()]
        if repos is None:
            names = sorted(set(names) | set(mem_repos))
        out = {}
        with conn.cursor() as cur:
            for repo in names:
                counts = {}
                for label, sql in [
                    ("files", "SELECT COUNT(*) FROM files WHERE repository_name=%s"),
                    ("business_rules", "SELECT COUNT(*) FROM business_rules b JOIN files f ON f.id=b.file_id WHERE f.repository_name=%s"),
                    ("key_points", "SELECT COUNT(*) FROM key_points WHERE repository_name=%s"),
                    ("classes", "SELECT COUNT(*) FROM classes c JOIN files f ON f.id=c.file_id WHERE f.repository_name=%s"),
                    ("functions", "SELECT COUNT(*) FROM functions fn JOIN files f ON f.id=fn.file_id WHERE f.repository_name=%s"),
                    ("bsg_specs", "SELECT COUNT(*) FROM bsg_specs b JOIN files f ON f.id=b.file_id WHERE f.repository_name=%s"),
                    ("translations", "SELECT COUNT(*) FROM translations WHERE repository_name=%s"),
                    ("agent_memory", "SELECT COUNT(*) FROM agent_memory WHERE repository_name=%s"),
                ]:
                    try:
                        cur.execute(sql, (repo,))
                        counts[label] = cur.fetchone()[0]
                    except Exception:
                        conn.rollback()          # table missing on an old schema
                        counts[label] = 0
                out[repo] = counts
        return out
    finally:
        conn.close()


def kuzu_status(repos: list[str] | None):
    from kb.graph.kuzu_store import get_reader, list_repos, summarize
    if not Path(_kuzu_path()).exists():
        return {}
    conn = get_reader(_kuzu_path())
    names = repos if repos is not None else list_repos(conn)
    return {r: summarize(conn, r) for r in names}


def chroma_collections(repos: list[str] | None) -> list[tuple[str, int]]:
    """[(collection_name, count)] belonging to Groundwork (optionally to some repos)."""
    path = _chroma_path()
    if not Path(path).exists():
        return []
    import chromadb
    client = chromadb.PersistentClient(path=path)
    # chromadb ≥0.6 returns names; older versions return Collection objects.
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if repos is None:
        wanted = [n for n in names if n.startswith(CHROMA_PREFIX)]
    else:
        targets = set()
        for r in repos:
            targets |= {f"{CHROMA_PREFIX}{_safe(r)}", f"{CHROMA_PREFIX}memory_{_safe(r)}"}
        wanted = [n for n in names if n in targets]
    return [(n, client.get_collection(n).count()) for n in sorted(wanted)]


def output_dirs(repos: list[str] | None) -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    if repos is None:
        return [OUTPUTS_DIR]
    return [OUTPUTS_DIR / r for r in repos if (OUTPUTS_DIR / r).exists()]


# ── Clearing ──────────────────────────────────────────────────────────────────

def clear_postgres(repos: list[str] | None, keep_memory: bool, include_keys: bool, drop: bool):
    from kb.relationaldb.initialize_db import (
        get_connection, clear_repo, clear_all, drop_all_tables, init_db)
    conn = get_connection()
    try:
        if repos is None:
            if drop:
                drop_all_tables(conn)
                with conn.cursor() as cur:
                    if not keep_memory:
                        cur.execute("DROP TABLE IF EXISTS agent_memory CASCADE")
                    if include_keys:
                        cur.execute("DROP TABLE IF EXISTS ai_providers CASCADE")
                conn.commit()
                conn.close()
                conn = None
                init_db()                         # re-create the schema empty
                print("  postgres: tables dropped and re-created")
                return
            clear_all(conn)
            with conn.cursor() as cur:
                if not keep_memory:
                    cur.execute("TRUNCATE TABLE agent_memory RESTART IDENTITY")
                if include_keys:
                    cur.execute("TRUNCATE TABLE ai_providers RESTART IDENTITY")
            conn.commit()
            print("  postgres: all repositories cleared"
                  + ("" if keep_memory else " (including agent memory)"))
            return

        for repo in repos:
            removed = clear_repo(conn, repo)      # files cascade to rules/classes/functions/bsg_specs
            mem = 0
            if not keep_memory:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM agent_memory WHERE repository_name=%s", (repo,))
                    mem = cur.rowcount
                conn.commit()
            print(f"  postgres: {repo} — {removed['files']} files, "
                  f"{removed['business_rules']} rules, {removed['key_points']} key points, "
                  f"{removed['classes']} classes, {removed['functions']} functions"
                  + ("" if keep_memory else f", {mem} memories"))
        if include_keys:
            print("  postgres: --include-keys ignored for --repo (keys aren't per-repo); use --all")
    finally:
        if conn is not None:
            conn.close()


def clear_kuzu(repos: list[str] | None, drop: bool):
    path = Path(_kuzu_path())
    if not path.exists():
        print("  kuzu: no database — nothing to clear")
        return
    if repos is None and drop:
        for p in (path, Path(str(path) + ".wal"), Path(str(path) + ".lock")):
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        print(f"  kuzu: deleted {path}")
        return
    from kb.graph.kuzu_store import get_connection, clear_repo, clear_all
    conn = get_connection(str(path))
    if repos is None:
        print(f"  kuzu: removed {clear_all(conn)} nodes")
    else:
        for repo in repos:
            print(f"  kuzu: {repo} — removed {clear_repo(conn, repo)} nodes")


def clear_chroma(repos: list[str] | None, keep_memory: bool, drop: bool):
    path = _chroma_path()
    if not Path(path).exists():
        print("  chroma: no database — nothing to clear")
        return
    if repos is None and drop and not keep_memory:
        shutil.rmtree(path)
        print(f"  chroma: deleted {path}")
        return
    import chromadb
    client = chromadb.PersistentClient(path=path)
    targets = chroma_collections(repos)
    if keep_memory:
        targets = [(n, c) for n, c in targets if not n.startswith(f"{CHROMA_PREFIX}memory_")]
    for name, count in targets:
        client.delete_collection(name)
        print(f"  chroma: deleted collection {name} ({count} entries)")
    if not targets:
        print("  chroma: no matching collections")


def clear_outputs(repos: list[str] | None):
    dirs = output_dirs(repos)
    for d in dirs:
        shutil.rmtree(d)
        print(f"  outputs: deleted {d}")
    if not dirs:
        print("  outputs: nothing to delete")


def clear_cache():
    if CACHE_DIR.exists():
        n = sum(1 for _ in CACHE_DIR.iterdir())
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  cache: removed {n} cached vectors")
    else:
        print("  cache: nothing to delete")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _report(repos, stores, keep_memory):
    """Prints what is stored now (= what would be removed). Returns False if
    there's nothing at all."""
    found = False
    print("\n  Currently stored:")
    if "postgres" in stores:
        try:
            for repo, counts in postgres_status(repos).items():
                shown = {k: v for k, v in counts.items()
                         if v and not (keep_memory and k == "agent_memory")}
                if shown:
                    found = True
                    print(f"    postgres  {repo:<20} " + ", ".join(f"{k} {v}" for k, v in shown.items()))
        except Exception as e:
            print(f"    postgres  unreachable ({e})")
    if "kuzu" in stores:
        try:
            for repo, s in kuzu_status(repos).items():
                if s["files"] or s["directories"]:
                    found = True
                    print(f"    kuzu      {repo:<20} {s['files']} files, {s['directories']} dirs, "
                          f"{s['depends_on']} DEPENDS_ON")
        except Exception as e:
            print(f"    kuzu      unreadable ({str(e).splitlines()[0]})")
    if "chroma" in stores:
        try:
            for name, count in chroma_collections(repos):
                if keep_memory and name.startswith(f"{CHROMA_PREFIX}memory_"):
                    continue
                found = True
                print(f"    chroma    {name:<20} {count} entries")
        except Exception as e:
            print(f"    chroma    unreadable ({e})")
    if "outputs" in stores:
        for d in output_dirs(repos):
            found = True
            print(f"    outputs   {d}")
    if "cache" in stores and repos is None and CACHE_DIR.exists() and any(CACHE_DIR.iterdir()):
        found = True
        print(f"    cache     {CACHE_DIR} ({sum(1 for _ in CACHE_DIR.iterdir())} files)")
    if not found:
        print("    (nothing)")
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description="Clear Groundwork's knowledge base",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("Usage:")[1])
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--repo", action="append", metavar="NAME",
                     help="Repository name as stored in the DBs (repeatable)")
    who.add_argument("--all", action="store_true", help="Every repository")
    p.add_argument("--only", default=None, help=f"Comma-separated stores from: {', '.join(STORES)}")
    p.add_argument("--keep-memory", action="store_true",
                   help="Keep agent memory (Postgres agent_memory + Chroma memory collections)")
    p.add_argument("--include-keys", action="store_true",
                   help="With --all, also delete stored API keys (ai_providers)")
    p.add_argument("--drop", action="store_true",
                   help="With --all: drop tables / delete DB files instead of emptying them, "
                        "then re-create the Postgres schema")
    p.add_argument("--dry-run", action="store_true", help="Show what would be removed, change nothing")
    p.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation")
    args = p.parse_args(argv)

    stores = STORES if not args.only else [s.strip() for s in args.only.split(",")]
    unknown = [s for s in stores if s not in STORES]
    if unknown:
        p.error(f"unknown store(s): {', '.join(unknown)}")
    if args.drop and not args.all:
        p.error("--drop only works with --all")
    repos = None if args.all else args.repo
    if repos is not None and "cache" in stores and args.only:
        print("  (cache isn't keyed by repo — it's only cleared with --all)")

    scope = "ALL repositories" if repos is None else ", ".join(repos)
    print(f"\n  Groundwork knowledge base — {scope}")
    print(f"  Stores: {', '.join(stores)}"
          + ("  (keeping agent memory)" if args.keep_memory else "")
          + ("  [DROP]" if args.drop else ""))
    found = _report(repos, stores, args.keep_memory)

    if args.dry_run:
        print("\n  Dry run — nothing changed.\n")
        return 0
    if not found and not args.drop:
        print("\n  Nothing to clear.\n")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("\n  Refusing to delete without confirmation in a non-interactive shell; pass -y.\n")
            return 1
        answer = input(f"\n  Type 'clear' to delete the above: ").strip()
        if answer != "clear":
            print("  Cancelled.\n")
            return 1

    print()
    failed = []
    steps = {
        "postgres": lambda: clear_postgres(repos, args.keep_memory, args.include_keys, args.drop),
        "kuzu": lambda: clear_kuzu(repos, args.drop),
        "chroma": lambda: clear_chroma(repos, args.keep_memory, args.drop),
        "outputs": lambda: clear_outputs(repos),
        "cache": lambda: clear_cache() if repos is None else None,
    }
    for store in stores:
        try:
            steps[store]()
        except Exception as e:
            failed.append(store)
            print(f"  ✗ {store}: {str(e).splitlines()[0]}")
    if failed:
        print(f"\n  Finished with errors in: {', '.join(failed)}\n")
        return 1
    print("\n  ✓ Done.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
