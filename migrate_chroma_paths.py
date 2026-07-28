#!/usr/bin/env python3
"""
Groundwork — ChromaDB path migration (one-off)

Rewrites the 'relative' metadata field so it starts at the repo name instead
of a full filesystem path — same fix as migrate_kuzu_paths.py /
migrate_postgres_paths.py, mirrored for ChromaDB.

ChromaDB ids can't be renamed in place. If your ids ARE the raw relative path
(a common pattern), this script detects that and does a delete + re-add with
the SAME embeddings and documents it already had, just a new id and corrected
metadata — nothing gets re-embedded, no OpenAI/model calls. If ids are
something else (e.g. a hash, or "<repo>::<relative>"), it updates the
metadata in place, which is cheaper and non-destructive.

Usage:
    python3 migrate_chroma_paths.py --repo nopCommerce --dry-run
    python3 migrate_chroma_paths.py --repo nopCommerce
"""

import argparse

CHROMA_DB_PATH = "./chroma_db"


def strip_to_repo(path: str, repo_name: str) -> str:
    """Keeps repo_name onward. Already-migrated or unrecognized paths pass through unchanged."""
    if path.startswith(f"{repo_name}/") or path == repo_name:
        return path
    marker = f"/{repo_name}/"
    idx = path.find(marker)
    if idx == -1:
        return path
    return path[idx + 1:]


def collection_name_for(repo_name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in (repo_name or "").lower())
    return f"groundwork_{safe}"


def plan_updates(ids, metadatas, embeddings, documents, repo_name):
    """
    Splits entries into two groups:
      id_rewrites   — the id itself is the old path, needs delete + re-add
      meta_only     — the id is independent of the path, just needs metadata updated
    Returns (id_rewrites, meta_only, unresolved).
    """
    id_rewrites, meta_only, unresolved = [], [], []

    for i, meta, emb, doc in zip(ids, metadatas, embeddings, documents):
        old_rel = (meta or {}).get("relative")
        if not old_rel:
            continue
        new_rel = strip_to_repo(old_rel, repo_name)
        if new_rel == old_rel:
            if not old_rel.startswith(f"{repo_name}/"):
                unresolved.append(old_rel)
            continue

        new_meta = dict(meta)
        new_meta["relative"] = new_rel

        if i == old_rel:
            id_rewrites.append({"old_id": i, "new_id": new_rel, "meta": new_meta,
                                "embedding": emb, "document": doc})
        else:
            meta_only.append({"id": i, "meta": new_meta})

    return id_rewrites, meta_only, unresolved


def main():
    ap = argparse.ArgumentParser(description="Rewrite ChromaDB relative paths to start at the repo name")
    ap.add_argument("--repo", required=True, help="Repository name")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = ap.parse_args()

    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    name = collection_name_for(args.repo)
    available = [c.name for c in client.list_collections()]
    if name not in available:
        print(f"  No collection '{name}'. Available: {available}")
        return
    col = client.get_collection(name)

    got = col.get(include=["metadatas", "embeddings", "documents"])
    id_rewrites, meta_only, unresolved = plan_updates(
        got["ids"], got["metadatas"], got["embeddings"], got["documents"], args.repo)
    total = len(id_rewrites) + len(meta_only)

    if args.dry_run:
        for u in id_rewrites:
            print(f"  [id+meta] {u['old_id']}\n    -> {u['new_id']}")
        for u in meta_only:
            print(f"  [meta only] {u['id']}: relative -> {u['meta']['relative']}")
        print(f"\n  {total} of {len(got['ids'])} entries would be updated "
              f"({len(id_rewrites)} id+metadata rewrite, {len(meta_only)} metadata only).")
    else:
        if meta_only:
            col.update(ids=[u["id"] for u in meta_only], metadatas=[u["meta"] for u in meta_only])
        if id_rewrites:
            col.delete(ids=[u["old_id"] for u in id_rewrites])
            col.add(
                ids=[u["new_id"] for u in id_rewrites],
                metadatas=[u["meta"] for u in id_rewrites],
                embeddings=[u["embedding"] for u in id_rewrites],
                documents=[u["document"] for u in id_rewrites],
            )
        print(f"\n  {total} of {len(got['ids'])} entries updated "
              f"({len(id_rewrites)} id+metadata rewrite, {len(meta_only)} metadata only).")

    if unresolved:
        print(f"  {len(unresolved)} 'relative' value(s) didn't contain '/{args.repo}/' — left unchanged:")
        for u in unresolved[:10]:
            print(f"    {u}")
        if len(unresolved) > 10:
            print(f"    ... and {len(unresolved) - 10} more")


if __name__ == "__main__":
    main()