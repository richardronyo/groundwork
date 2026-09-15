"""
Groundwork — Agent memory (Postgres record + ChromaDB semantic recall)

Two stores, same content, for two different access patterns:
  - Postgres `agent_memory` table: the durable, queryable record (by repo,
    file, or memory_type — e.g. "show me every correction made on this file").
  - A dedicated Chroma collection per repo ("groundwork_memory_<repo>"),
    kept separate from the "groundwork_<repo>" business-rules collection in
    embeddings.py, because that one's vector *dimension* is tied to the
    repo's key-point count (see collection_name_for() there) and isn't a
    general-purpose embedding space — memory needs plain semantic search
    over free-form text, not similarity-to-key-points.

Reuses encode_texts() and get_collection() from kb.vector.embeddings rather
than re-implementing model loading or Chroma setup, so memory embeddings
stay on the same model/versioning as the rest of the project.
"""

from functools import lru_cache

from kb.relationaldb.initialize_db import get_connection
from kb.vector.embeddings import get_collection, EMBED_MODEL, CHROMA_DB_PATH

MEMORY_TYPES = {"fact", "reflection", "correction"}


def _memory_collection_name(repo_name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in repo_name.lower())
    return f"groundwork_memory_{safe}"


@lru_cache(maxsize=1)
def _embed_model():
    """
    encode_texts() in embeddings.py loads a fresh SentenceTransformer on every
    call — fine for a one-shot batch embedding job, but remember()/recall()
    are meant to be called many times inside an agent loop, so that cost adds
    up fast. Cache the loaded model here instead of going through
    encode_texts() for every single call.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def _embed(texts: list[str]):
    return _embed_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )


def remember(repo_name: str, content: str, memory_type: str = "fact",
             file_path: str | None = None) -> int:
    """
    Stores one memory entry in both Postgres (durable record) and Chroma
    (semantic recall). Returns the Postgres row id.
    """
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"memory_type must be one of {MEMORY_TYPES}, got {memory_type!r}")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO agent_memory (repository_name, file_path, memory_type, content)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (repo_name, file_path, memory_type, content)
            )
            memory_id = cur.fetchone()[0]
            conn.commit()
    finally:
        conn.close()

    vector = _embed([content])[0]
    collection = get_collection(CHROMA_DB_PATH, _memory_collection_name(repo_name))
    collection.upsert(
        ids=[f"mem_{memory_id}"],
        embeddings=[vector.tolist()],
        documents=[content],
        metadatas=[{
            "memory_type": memory_type,
            "file_path": file_path or "",
        }],
    )
    return memory_id


def recall(repo_name: str, query: str, n_results: int = 5,
           memory_type: str | None = None) -> list[str]:
    """
    Semantic search over this repo's memories. Returns matching memory texts,
    most relevant first. Optionally filter to one memory_type.
    """
    collection = get_collection(CHROMA_DB_PATH, _memory_collection_name(repo_name))
    if collection.count() == 0:
        return []

    vector = _embed([query])[0]
    where = {"memory_type": memory_type} if memory_type else None
    results = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=min(n_results, collection.count()),
        where=where,
    )
    return results["documents"][0] if results["documents"] else []


def recall_for_file(repo_name: str, file_path: str, n_results: int = 5) -> list[str]:
    """Memories previously recorded against this specific file (exact match,
    not semantic — useful before re-processing a file that's been visited
    before, e.g. on a --only-unprocessed re-run)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT content FROM agent_memory
                   WHERE repository_name = %s AND file_path = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (repo_name, file_path, n_results)
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()