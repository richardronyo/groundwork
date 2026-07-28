#!/usr/bin/env python3
"""
Groundwork — Kùzu path migration (one-off)

Rewrites File.relative so it starts at the repo name instead of carrying the
full filesystem path.

    Before: /Users/richardlui/School/groundwork/nopCommerce/src/Libraries/Nop.Core/Domain/Orders/Order.cs
    After:  nopCommerce/src/Libraries/Nop.Core/Domain/Orders/Order.cs

DEPENDS_ON / CONTAINS / SAME_DIR edges are untouched by this — Kùzu relationships
reference nodes by internal ID, not by property value, so renaming `relative`
doesn't break any existing edge.

Idempotent: paths that already start with "<repo>/" are left alone, so it's
safe to re-run (e.g. after fixing the pipeline and re-ingesting a repo that
was already migrated).

Usage:
    python3 migrate_kuzu_paths.py --repo nopCommerce --dry-run   # preview
    python3 migrate_kuzu_paths.py --repo nopCommerce             # apply
    python3 migrate_kuzu_paths.py --repo flask
"""

import argparse


def strip_to_repo(path: str, repo_name: str) -> str:
    """Keeps repo_name onward. Already-migrated or unrecognized paths pass through unchanged."""
    if path.startswith(f"{repo_name}/") or path == repo_name:
        return path                      # already migrated
    marker = f"/{repo_name}/"
    idx = path.find(marker)
    if idx == -1:
        return path                      # doesn't contain the repo name — leave it, flag it
    return path[idx + 1:]                # drop the leading '/', keep "<repo>/..." onward


def main():
    ap = argparse.ArgumentParser(description="Rewrite Kùzu File.relative to start at the repo name")
    ap.add_argument("--repo", required=True, help="Repository name")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = ap.parse_args()

    from kb.graph.kuzu_store import get_connection, rows_to_dicts

    conn = get_connection()
    rows = rows_to_dicts(conn.execute(
        "MATCH (f:File {repository_name: $repo}) RETURN f.relative",
        {"repo": args.repo},
    ))
    paths = [r[0] for r in rows]

    if not paths:
        print(f"  No File nodes found for repository_name='{args.repo}'. Check the name and try again.")
        return

    changed, unresolved = 0, []
    for old in paths:
        new = strip_to_repo(old, args.repo)
        if new == old:
            if not old.startswith(f"{args.repo}/"):
                unresolved.append(old)
            continue
        changed += 1
        if args.dry_run:
            print(f"  {old}\n    -> {new}")
        else:
            conn.execute(
                "MATCH (f:File {repository_name: $repo, relative: $old}) SET f.relative = $new",
                {"repo": args.repo, "old": old, "new": new},
            )

    verb = "would be" if args.dry_run else "were"
    print(f"\n  {changed} of {len(paths)} paths {verb} updated.")
    if unresolved:
        print(f"  {len(unresolved)} path(s) didn't contain '/{args.repo}/' — left unchanged:")
        for u in unresolved[:10]:
            print(f"    {u}")
        if len(unresolved) > 10:
            print(f"    ... and {len(unresolved) - 10} more")


if __name__ == "__main__":
    main()