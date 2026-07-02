"""
Groundwork — Stage 5: ChromaDB Vector Store (PostgreSQL-backed)

Reads business rules and key points from PostgreSQL. For each file builds a
(n_keypoints,) vector where dimension k = average BERTScore F1 of the file's
rules against key point k. Stores vectors + metadata in ChromaDB.

Usage:
    python3 embeddings.py --repo flask
    python3 embeddings.py            # if only one repo in the DB

Requirements:
    pip install chromadb psycopg python-dotenv bert-score torch transformers
"""

import json
import sys
import argparse

import chromadb
from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection, load_business_rules_from_db,
    load_key_points_from_db, list_repositories,
)

load_dotenv()

CHROMA_COLLECTION = "groundwork"  # base name; per-repo becomes groundwork_<repo>


def collection_name_for(repo_name: str) -> str:
    """Per-repo collection name. Each repo has its own collection because the
    vector dimension (= key-point count) differs per repo, and ChromaDB locks
    dimension per collection."""
    safe = "".join(c if c.isalnum() else "_" for c in repo_name.lower())
    return f"{CHROMA_COLLECTION}_{safe}"
CHROMA_DB_PATH    = "./chroma_db"
BERT_MODEL        = "distilbert-base-uncased"


def load_bert_scorer():
    try:
        from bert_score import score as bert_score_fn
        return bert_score_fn
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        sys.exit(1)


def compute_file_vector(rules, key_points, bert_score_fn) -> list[float]:
    file_vector = []
    for kp in key_points:
        refs = [kp] * len(rules)
        _, _, F1 = bert_score_fn(cands=rules, refs=refs,
                                 model_type=BERT_MODEL, verbose=False)
        file_vector.append(round(float(F1.mean()), 6))
    return file_vector


def get_collection(db_path, collection_name, expected_dim=None) -> chromadb.Collection:
    """
    Returns the collection, recreating it if its stored vector dimension
    no longer matches expected_dim. ChromaDB locks a collection to the
    dimension of its first insert, so a changed key-point count (which
    changes the vector length) requires a fresh collection.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})

    if expected_dim is not None and collection.count() > 0:
        # Peek at one stored vector to read its dimension
        existing = collection.peek(1)
        embeddings = existing.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            stored_dim = len(embeddings[0])
            if stored_dim != expected_dim:
                print(f"  ⚠ Dimension changed ({stored_dim} → {expected_dim}); "
                      f"recreating collection '{collection_name}'.")
                client.delete_collection(collection_name)
                collection = client.get_or_create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"})

    return collection


def build_vectorstore(business_rules, key_points, collection, bert_score_fn):
    total = len(business_rules)
    n_kp  = len(key_points)
    print(f"\n  Files to process  : {total}")
    print(f"  Key points        : {n_kp}")
    print(f"  Vector dimensions : {n_kp} (one per key point)\n")

    upserted = skipped = 0
    for i, (relative, rules) in enumerate(business_rules.items()):
        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        name = relative.split("/")[-1]
        print(f"\r  [{bar}] {i+1}/{total}  {name:<40}", end="", flush=True)

        if not rules:
            skipped += 1
            continue

        file_vector = compute_file_vector(rules, key_points, bert_score_fn)
        top_index = int(max(range(n_kp), key=lambda k: file_vector[k]))
        metadata = {
            "relative": relative, "name": name, "rule_count": len(rules),
            "top_kp": key_points[top_index], "top_kp_index": top_index,
            "top_score": round(file_vector[top_index], 4),
            "kp_scores": json.dumps({f"kp_{k}": file_vector[k] for k in range(n_kp)}),
        }
        collection.upsert(
            ids=[relative], embeddings=[file_vector],
            metadatas=[metadata], documents=["\n\n".join(rules)])
        upserted += 1

    print(f"\n\n  ✓ {upserted} files added to ChromaDB, {skipped} skipped (no rules).")


def print_summary(collection, key_points):
    count = collection.count()
    print(f"\n  ┌─ ChromaDB Summary ───────────────────────────────┐")
    print(f"  │  Collection       : {collection.name:<30}│")
    print(f"  │  File vectors     : {count:<30}│")
    print(f"  │  Vector dimensions: {len(key_points):<30}│")
    print(f"  └──────────────────────────────────────────────────┘\n")


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        print("Error: no repositories in DB. Run metadata.py first.")
        sys.exit(1)
    if requested:
        return requested
    if len(repos) == 1:
        print(f"  Using only repository in DB: {repos[0]}")
        return repos[0]
    print("  Multiple repositories — specify one with --repo:")
    for r in repos:
        print(f"    - {r}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — build ChromaDB vectors from PostgreSQL")
    parser.add_argument("--repo", help="Repository name as stored in the DB")
    parser.add_argument("--collection", default=None,
                        help="Override collection name (default: groundwork_<repo>)")
    parser.add_argument("--db", default=CHROMA_DB_PATH)
    args = parser.parse_args()

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
        print(f"\n  Loading data for '{repo_name}' from PostgreSQL...")
        business_rules = load_business_rules_from_db(conn, repo_name)
        key_points     = load_key_points_from_db(conn, repo_name)

        if not business_rules:
            print(f"Error: no business rules for '{repo_name}'.")
            sys.exit(1)
        if not key_points:
            print(f"Error: no key points for '{repo_name}'. Run synthesize.py first.")
            sys.exit(1)

        print(f"  Loaded {len(business_rules)} files with rules")
        print(f"  Loaded {len(key_points)} key points")
        print(f"  ChromaDB path      : {args.db}")
        print(f"  BERTScore model    : {BERT_MODEL}")
    finally:
        conn.close()

    bert_score_fn = load_bert_scorer()
    print(f"  ✓ BERTScore model loaded.\n")

    coll_name = args.collection or collection_name_for(repo_name)
    collection = get_collection(args.db, coll_name, expected_dim=len(key_points))
    build_vectorstore(business_rules, key_points, collection, bert_score_fn)
    print_summary(collection, key_points)


if __name__ == "__main__":
    main()