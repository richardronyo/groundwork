#!/usr/bin/env python3
"""
Groundwork — PostgreSQL path migration (one-off)

Rewrites files.file_path so it starts at the repo name instead of carrying
the full filesystem path — the same fix as migrate_kuzu_paths.py, mirrored
for Postgres.

    Before: /Users/richardlui/School/groundwork/nopCommerce/src/Libraries/Nop.Core/Domain/Orders/Order.cs
    After:  nopCommerce/src/Libraries/Nop.Core/Domain/Orders/Order.cs

I've only confirmed the bug lives in `files.file_path` (from your Kùzu paste
and the schema in memory — files.repository_name / files.file_path). If any
other table stores a path directly (rather than a file_id foreign key), pass
--scan-other-tables and this will look for TEXT/VARCHAR columns named like
'%path%' that sit alongside a repository_name column, and migrate those too.

Uses kb.relationaldb.initialize_db.get_connection() — same connection the
rest of the pipeline uses.

Usage:
    python3 migrate_postgres_paths.py --repo nopCommerce --dry-run
    python3 migrate_postgres_paths.py --repo nopCommerce
    python3 migrate_postgres_paths.py --repo nopCommerce --scan-other-tables
"""

import argparse


def strip_to_repo(path: str, repo_name: str) -> str:
    """Keeps repo_name onward. Already-migrated or unrecognized paths pass through unchanged."""
    if path.startswith(f"{repo_name}/") or path == repo_name:
        return path
    marker = f"/{repo_name}/"
    idx = path.find(marker)
    if idx == -1:
        return path
    return path[idx + 1:]


def migrate_files_table(conn, repo: str, dry_run: bool):
    with conn.cursor() as cur:
        cur.execute("SELECT id, file_path FROM files WHERE repository_name = %s", (repo,))
        rows = cur.fetchall()

    if not rows:
        print(f"  No rows in 'files' for repository_name='{repo}'.")
        return

    changed, unresolved = 0, []
    with conn.cursor() as cur:
        for file_id, old in rows:
            new = strip_to_repo(old, repo)
            if new == old:
                if not old.startswith(f"{repo}/"):
                    unresolved.append(old)
                continue
            changed += 1
            if dry_run:
                print(f"  {old}\n    -> {new}")
            else:
                cur.execute("UPDATE files SET file_path = %s WHERE id = %s", (new, file_id))
    if not dry_run:
        conn.commit()

    verb = "would be" if dry_run else "were"
    print(f"\n  files.file_path: {changed} of {len(rows)} paths {verb} updated.")
    if unresolved:
        print(f"  {len(unresolved)} path(s) didn't contain '/{repo}/' — left unchanged:")
        for u in unresolved[:10]:
            print(f"    {u}")
        if len(unresolved) > 10:
            print(f"    ... and {len(unresolved) - 10} more")


def find_other_path_columns(conn):
    """Tables/columns that look like they might also store a path, scoped by
    a sibling repository_name column so a row can be safely identified and rewritten."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.table_name, c.column_name
            FROM information_schema.columns c
            WHERE c.table_schema = 'public'
              AND c.data_type IN ('text', 'character varying')
              AND c.column_name ILIKE '%path%'
              AND c.table_name != 'files'
              AND EXISTS (
                  SELECT 1 FROM information_schema.columns c2
                  WHERE c2.table_schema = 'public' AND c2.table_name = c.table_name
                    AND c2.column_name = 'repository_name'
              )
        """)
        return cur.fetchall()


def migrate_other_table(conn, table: str, column: str, repo: str, dry_run: bool):
    # ctid identifies a row within this run; every ordinary Postgres table has
    # one regardless of its primary key, which we don't know for arbitrary tables.
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT ctid, {column} FROM {table} WHERE repository_name = %s AND {column} IS NOT NULL",
            (repo,),
        )
        rows = cur.fetchall()

    if not rows:
        return

    changed = 0
    with conn.cursor() as cur:
        for ctid, old in rows:
            new = strip_to_repo(old, repo)
            if new == old:
                continue
            changed += 1
            if dry_run:
                print(f"  [{table}.{column}] {old}\n    -> {new}")
            else:
                cur.execute(f"UPDATE {table} SET {column} = %s WHERE ctid = %s", (new, ctid))
    if not dry_run:
        conn.commit()

    verb = "would be" if dry_run else "were"
    print(f"  {table}.{column}: {changed} of {len(rows)} paths {verb} updated.")


def main():
    ap = argparse.ArgumentParser(description="Rewrite Postgres file paths to start at the repo name")
    ap.add_argument("--repo", required=True, help="Repository name")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    ap.add_argument("--scan-other-tables", action="store_true",
                    help="Also look for path-like columns outside 'files' and migrate those too")
    args = ap.parse_args()

    from kb.relationaldb.initialize_db import get_connection
    conn = get_connection()

    migrate_files_table(conn, args.repo, args.dry_run)

    if args.scan_other_tables:
        others = find_other_path_columns(conn)
        if not others:
            print("\n  No other path-like columns found alongside a repository_name column.")
        for table, column in others:
            migrate_other_table(conn, table, column, args.repo, args.dry_run)

    conn.close()


if __name__ == "__main__":
    main()