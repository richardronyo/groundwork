"""ChromaDB store: business-rule embeddings for semantic search.

One collection per repo, named "groundwork_<repo>". Each business rule is one
document, so a search returns individual rules rather than whole files.

    document  "<feature>: <rule>"
    id        "<file_id>::rule::<business_rules.id>"   (links back to Postgres)
    metadata  repo, file_id, rel_path, language, feature, rule_id, model

PostgreSQL stays the source of truth: this collection is rebuilt from the
business_rules table, so it can always be thrown away and re-embedded.

Embeddings use sentence-transformers (all-MiniLM-L6-v2 by default) with cosine
distance. Pass a different embedding_function to VectorStore to change that.
"""

from __future__ import annotations

import re

import chromadb

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def collection_name(repo: str) -> str:
    """Chroma names allow [A-Za-z0-9._-] and must start and end alphanumeric."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", repo).strip("._-") or "repo"
    return f"groundwork_{safe}"


def rule_document(feature: str, rule: str) -> str:
    """The text that gets embedded for one rule."""
    return f"{feature}: {rule}"


class VectorStore:
    def __init__(self, path: str = "./chroma_db", embedding_function=None):
        self.client = chromadb.PersistentClient(path=path)
        if embedding_function is None:
            # Imported here so the module loads without sentence-transformers
            # installed, e.g. when a test passes its own embedding function.
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction)
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=DEFAULT_EMBEDDING_MODEL)
        self.embedding_function = embedding_function

    def collection(self, repo: str):
        return self.client.get_or_create_collection(
            name=collection_name(repo),
            embedding_function=self.embedding_function,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def delete_repo(self, repo: str) -> None:
        try:
            self.client.delete_collection(collection_name(repo))
        except Exception:
            pass  # nothing stored for this repo yet

    def add_business_rules(self, repo: str, rules: list[dict],
                           batch_size: int = 256) -> int:
        """Embed rule rows as returned by RelationalStore.fetch_business_rules().

        Returns how many rules were stored.
        """
        if not rules:
            return 0
        collection = self.collection(repo)
        for start in range(0, len(rules), batch_size):
            batch = rules[start:start + batch_size]
            collection.add(
                ids=[f"{r['file_id']}::rule::{r['id']}" for r in batch],
                documents=[rule_document(r["feature"], r["rule"]) for r in batch],
                metadatas=[{
                    "repo": repo, "file_id": r["file_id"],
                    "rel_path": r["rel_path"], "language": r["language"],
                    "feature": r["feature"], "rule_id": r["id"],
                    "model": r["model"],
                } for r in batch],
            )
        return len(rules)

    def search_rules(self, repo: str, text: str, n_results: int = 5,
                     file_id: str | None = None) -> list[dict]:
        """Rules closest in meaning to text, optionally within one file."""
        result = self.collection(repo).query(
            query_texts=[text], n_results=n_results,
            where={"file_id": file_id} if file_id else None,
        )
        return [
            {"id": id_, "document": doc, "distance": dist, **meta}
            for id_, doc, dist, meta in zip(result["ids"][0], result["documents"][0],
                                            result["distances"][0], result["metadatas"][0])
        ]

    def count(self, repo: str) -> int:
        return self.collection(repo).count()