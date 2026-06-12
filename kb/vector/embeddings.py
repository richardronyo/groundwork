"""
Groundwork — ChromaDB Vector Store Builder

For each file:
  1. Embeds each business rule using OpenAI embeddings
  2. Averages the rule vectors to produce a single file vector
  3. Stores the file vector + metadata in a local ChromaDB collection

Usage:
    python3 build_vectorstore.py
    python3 build_vectorstore.py --rules business_rules.json --keypoints repo_function.json
    python3 build_vectorstore.py --collection groundwork --db ./chroma_db

Requirements:
    pip install openai chromadb python-dotenv
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"

CHROMA_COLLECTION = "groundwork"
CHROMA_DB_PATH = "./chroma_db"

RETRY_DELAY = 1

# ── OpenAI Embedding Client ───────────────────────────────────────────────────


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(
    client: OpenAI,
    texts: list[str],
    retries: int = 3,
) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI embeddings.
    """

    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )

            return [
                item.embedding
                for item in response.data
            ]

        except Exception as e:
            if attempt < retries - 1:
                print(
                    f"\n  Embedding API error: {e}"
                    f" — retrying in {RETRY_DELAY}s..."
                )
                time.sleep(RETRY_DELAY)
            else:
                raise


# ── Vector Math ───────────────────────────────────────────────────────────────


def average_vectors(vectors: list[list[float]]) -> list[float]:
    """
    Element-wise average of vectors.
    """

    if not vectors:
        return []

    dim = len(vectors[0])
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    count = len(vectors)

    return [
        value / count
        for value in result
    ]


# ── ChromaDB Setup ────────────────────────────────────────────────────────────


def get_collection(
    db_path: str,
    collection_name: str,
) -> chromadb.Collection:

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
    client: OpenAI,
    collection: chromadb.Collection,
):
    total = len(business_rules)

    print(f"\n  Building vectors for {total} files...\n")

    upserted = 0
    skipped = 0

    for i, (relative, rules) in enumerate(
        business_rules.items()
    ):
        pct = int((i + 1) / total * 40)
        bar = "█" * pct + "░" * (40 - pct)

        name = relative.split("/")[-1]

        print(
            f"\r  [{bar}] {i+1}/{total}  {name:<40}",
            end="",
            flush=True,
        )

        if not rules:
            skipped += 1
            continue

        # Embed all business rules for this file
        rule_vectors = embed_texts(
            client,
            rules,
        )

        # Average rule vectors → single file vector
        file_vector = average_vectors(
            rule_vectors
        )

        if not file_vector:
            skipped += 1
            continue

        metadata = {
            "relative": relative,
            "name": name,
            "rule_count": len(rules),
        }

        collection.upsert(
            ids=[relative],
            embeddings=[file_vector],
            metadatas=[metadata],
            documents=[
                "\n\n".join(rules)
            ],
        )

        upserted += 1

    print(
        f"\n\n  ✓ {upserted} files added to ChromaDB, "
        f"{skipped} skipped (no rules)."
    )


def print_summary(
    collection: chromadb.Collection,
    key_points: list[str],
):
    count = collection.count()

    print(
        "\n  ┌─ ChromaDB Summary "
        "───────────────────────────────┐"
    )
    print(
        f"  │  Collection     : "
        f"{collection.name:<32}│"
    )
    print(
        f"  │  File vectors   : "
        f"{count:<32}│"
    )
    print(
        f"  │  Key points     : "
        f"{len(key_points):<32}│"
    )
    print(
        "  └──────────────────────────────────────────────────┘"
    )

    print("\n  Example query (Python):")

    print(
        """
    import chromadb

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    col = client.get_collection(
        "groundwork"
    )

    results = col.query(
        query_texts=[
            "authentication and user login"
        ],
        n_results=5
    )

    for r in results["metadatas"][0]:
        print(r["relative"])
    """
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Groundwork — build ChromaDB vector "
            "store from business rules"
        )
    )

    parser.add_argument(
        "--rules",
        default="business_rules.json",
        help=(
            "Path to business_rules.json "
            "(default: business_rules.json)"
        ),
    )

    parser.add_argument(
        "--keypoints",
        default="repo_function.json",
        help=(
            "Path to repo_function.json "
            "(default: repo_function.json)"
        ),
    )

    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION,
        help=(
            f"ChromaDB collection name "
            f"(default: {CHROMA_COLLECTION})"
        ),
    )

    parser.add_argument(
        "--db",
        default=CHROMA_DB_PATH,
        help=(
            f"ChromaDB persistence path "
            f"(default: {CHROMA_DB_PATH})"
        ),
    )

    args = parser.parse_args()

    rules_path = Path(args.rules)
    kp_path = Path(args.keypoints)

    if not rules_path.exists():
        print(
            f"Error: '{rules_path}' not found. "
            "Run extract_rules.py first."
        )
        sys.exit(1)

    if not kp_path.exists():
        print(
            f"Error: '{kp_path}' not found. "
            "Run extract_rules.py first."
        )
        sys.exit(1)

    with open(rules_path) as f:
        business_rules = json.load(f)

    with open(kp_path) as f:
        key_points = json.load(f)

    print(
        f"\n  Loaded {len(business_rules)} "
        f"files from {rules_path}"
    )

    print(
        f"  Loaded {len(key_points)} "
        f"key points from {kp_path}"
    )

    print(f"  ChromaDB path      : {args.db}")
    print(f"  ChromaDB collection: {args.collection}")
    print(f"  Embedding model    : {EMBEDDING_MODEL}")

    client = get_client()

    collection = get_collection(
        args.db,
        args.collection,
    )

    build_vectorstore(
        business_rules,
        key_points,
        client,
        collection,
    )

    print_summary(
        collection,
        key_points,
    )


if __name__ == "__main__":
    main()