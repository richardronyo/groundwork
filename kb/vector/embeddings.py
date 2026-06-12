"""
Groundwork — ChromaDB Vector Store Builder

For each file:
  1. Computes BERTScore F1 for every (rule, key_point) pair → (n_rules, n_keypoints) matrix
  2. Averages across rules → (1, n_keypoints) vector
  3. Each dimension in the vector represents one key point
  4. Stores the vector + metadata in ChromaDB

Usage:
    python3 embeddings.py
    python3 embeddings.py --rules business_rules.json --keypoints repo_function.json
    python3 embeddings.py --collection groundwork --db ./chroma_db

Requirements:
    pip install chromadb python-dotenv bert-score torch transformers
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_COLLECTION = "groundwork"
CHROMA_DB_PATH    = "./chroma_db"
BERT_MODEL        = "distilbert-base-uncased"


# ── BERTScore ─────────────────────────────────────────────────────────────────

def load_bert_scorer():
    try:
        from bert_score import score as bert_score_fn
        return bert_score_fn
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        sys.exit(1)


def compute_file_vector(
    rules: list[str],
    key_points: list[str],
    bert_score_fn,
) -> list[float]:
    """
    Builds a (n_keypoints,) vector for a file.

    For each key point k:
      - Score every rule against k  →  [f1_rule1, f1_rule2, ...]
      - Average those scores        →  one float for dimension k

    Result: a vector where dimension k = how much this file's rules
            collectively relate to key point k.
    """
    n_kp = len(key_points)
    file_vector = []

    for kp in key_points:
        # Score all rules against this one key point
        # cands = rules, refs = same key point repeated for each rule
        refs = [kp] * len(rules)
        _, _, F1 = bert_score_fn(
            cands=rules,
            refs=refs,
            model_type=BERT_MODEL,
            verbose=False,
        )
        # Average F1 across all rules for this key point
        avg_f1 = float(F1.mean())
        file_vector.append(round(avg_f1, 6))

    return file_vector


# ── ChromaDB Setup ────────────────────────────────────────────────────────────

def get_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_vectorstore(
    business_rules: dict[str, list[str]],
    key_points: list[str],
    collection: chromadb.Collection,
    bert_score_fn,
):
    total  = len(business_rules)
    n_kp   = len(key_points)

    print(f"\n  Files to process : {total}")
    print(f"  Key points       : {n_kp}")
    print(f"  Vector dimensions: {n_kp}  (one per key point)\n")

    upserted = 0
    skipped  = 0

    for i, (relative, rules) in enumerate(business_rules.items()):
        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        name = relative.split("/")[-1]
        print(f"\r  [{bar}] {i+1}/{total}  {name:<40}", end="", flush=True)

        if not rules:
            skipped += 1
            continue

        # Build the (n_keypoints,) vector for this file
        file_vector = compute_file_vector(rules, key_points, bert_score_fn)

        # Find which key point this file is most aligned with
        top_index = int(max(range(n_kp), key=lambda k: file_vector[k]))
        top_score = file_vector[top_index]
        top_kp    = key_points[top_index]

        metadata = {
            "relative":    relative,
            "name":        name,
            "rule_count":  len(rules),
            "top_kp":      top_kp,
            "top_kp_index":top_index,
            "top_score":   round(top_score, 4),
            # Store all scores as JSON string for notebook inspection
            "kp_scores":   json.dumps(
                {f"kp_{k}": file_vector[k] for k in range(n_kp)}
            ),
        }

        collection.upsert(
            ids=[relative],
            embeddings=[file_vector],
            metadatas=[metadata],
            documents=["\n\n".join(rules)],
        )
        upserted += 1

    print(f"\n\n  ✓ {upserted} files added to ChromaDB, {skipped} skipped (no rules).")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    collection: chromadb.Collection,
    key_points: list[str],
):
    count   = collection.count()
    results = collection.get(include=["metadatas"])

    print(f"\n  ┌─ ChromaDB Summary ───────────────────────────────┐")
    print(f"  │  Collection      : {collection.name:<31}│")
    print(f"  │  File vectors    : {count:<31}│")
    print(f"  │  Vector dimensions: {len(key_points):<30}│")
    print(f"  └──────────────────────────────────────────────────┘")

    # For each key point, show the top 3 most aligned files
    print("\n  Top files per key point:\n")
    for k, kp in enumerate(key_points):
        # Sort files by their score for this key point
        scored = []
        for meta in results["metadatas"]:
            scores = json.loads(meta.get("kp_scores", "{}"))
            score  = scores.get(f"kp_{k}", 0.0)
            scored.append((score, meta["name"]))
        scored.sort(reverse=True)

        kp_short = kp[:70] + "..." if len(kp) > 70 else kp
        print(f"  KP {k+1:02d}: {kp_short}")
        for score, name in scored[:3]:
            print(f"         [{score:.4f}]  {name}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — build key-point-aligned ChromaDB vectors using BERTScore"
    )
    parser.add_argument("--rules",      default="business_rules.json")
    parser.add_argument("--keypoints",  default="repo_function.json")
    parser.add_argument("--collection", default=CHROMA_COLLECTION)
    parser.add_argument("--db",         default=CHROMA_DB_PATH)
    args = parser.parse_args()

    rules_path = Path(args.rules)
    kp_path    = Path(args.keypoints)

    if not rules_path.exists():
        print(f"Error: '{rules_path}' not found. Run extract_rules.py first.")
        sys.exit(1)
    if not kp_path.exists():
        print(f"Error: '{kp_path}' not found. Run extract_rules.py first.")
        sys.exit(1)

    with open(rules_path) as f:
        business_rules = json.load(f)
    with open(kp_path) as f:
        key_points = json.load(f)

    print(f"\n  Loaded {len(business_rules)} files from {rules_path}")
    print(f"  Loaded {len(key_points)} key points from {kp_path}")
    print(f"  ChromaDB path      : {args.db}")
    print(f"  ChromaDB collection: {args.collection}")
    print(f"  BERTScore model    : {BERT_MODEL}")
    print(f"  Vector dimensions  : {len(key_points)} (one per key point)")

    bert_score_fn = load_bert_scorer()
    print(f"  ✓ BERTScore model loaded.\n")

    collection = get_collection(args.db, args.collection)

    build_vectorstore(business_rules, key_points, collection, bert_score_fn)
    print_summary(collection, key_points)


if __name__ == "__main__":
    main()