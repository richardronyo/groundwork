#!/usr/bin/env python3
"""
Groundwork — Knowledge Base API

Plain getters and setters for everything in the knowledge base. Each function
opens and closes its own connection, so callers never pass one around.

    PostgreSQL  files, business rules, key points
    Kùzu        file dependencies (DEPENDS_ON)
    ChromaDB    key-point-aligned vectors

Usage:
    from kb.api import get_key_points, get_rules, get_dependents

    for kp in get_key_points("flask"):
        print(kp)

    rules = get_rules("flask", "src/flask/app.py")
    who   = get_dependents("flask", "src/flask/app.py")

Every function takes `repo` first. Getters return empty (None / [] / {}) rather
than raising when something isn't there.

Requirements:
    pip install psycopg kuzu chromadb python-dotenv
"""

from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection as _pg,
    save_file as _save_file,
    save_business_rules as _save_rules,
    save_key_points as _save_key_points,
    get_files as _get_files,
    load_business_rules_from_db as _load_rules,
    load_key_points_from_db as _load_key_points,
    list_repositories as _list_repos_pg,
    get_repo_status as _repo_status,
    clear_repo as _pg_clear_repo,
)

load_dotenv()

CHROMA_DB_PATH    = "./chroma_db"
CHROMA_COLLECTION = "groundwork"

METRIC_FIELDS = ("classes", "functions", "methods",
                 "async_functions", "imports", "lines")


# ── Repositories ──────────────────────────────────────────────────────────────

def list_repos() -> list:
    """Every repository name in the knowledge base."""
    conn = _pg()
    try:
        return _list_repos_pg(conn)
    finally:
        conn.close()


def get_repo(repo: str) -> dict:
    """Summary of one repository across all three stores."""
    conn = _pg()
    try:
        status = _repo_status(conn, repo)
    finally:
        conn.close()

    out = dict(status) if status else {"repository_name": repo}
    out["dependencies"] = _count_edges(repo)
    out["vectors"] = _count_vectors(repo)
    return out


def delete_repo(repo: str) -> dict:
    """Removes a repository from all three stores. Returns what was removed."""
    removed = {}

    conn = _pg()
    try:
        removed["postgres"] = _pg_clear_repo(conn, repo)
    finally:
        conn.close()

    try:
        from kb.graph.kuzu_store import get_connection as _kz, clear_repo as _kz_clear
        removed["kuzu_nodes"] = _kz_clear(_kz(), repo)
    except Exception:
        removed["kuzu_nodes"] = 0

    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(collection_name(repo))
        removed["chroma_collection"] = collection_name(repo)
    except Exception:
        removed["chroma_collection"] = None

    return removed


# ── Files (PostgreSQL) ────────────────────────────────────────────────────────

def get_files(repo: str, only_unprocessed: bool = False) -> list:
    """All files in a repo: [{file_id, file_path, language, rules_extracted}]."""
    conn = _pg()
    try:
        return _get_files(conn, repo, only_unprocessed=only_unprocessed)
    finally:
        conn.close()


def get_file(repo: str, path: str) -> dict:
    """One file's metrics, or None if it isn't in the knowledge base."""
    conn = _pg()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_path, language, classes, functions, methods,
                       async_functions, imports, lines, rules_extracted
                FROM files
                WHERE repository_name = %s AND file_path = %s
            """, (repo, path))
            r = cur.fetchone()
    finally:
        conn.close()

    if not r:
        return None
    return {"file_id": r[0], "file_path": r[1], "language": r[2],
            "classes": r[3], "functions": r[4], "methods": r[5],
            "async_functions": r[6], "imports": r[7], "lines": r[8],
            "rules_extracted": r[9]}


def set_file(repo: str, path: str, language: str, **metrics) -> int:
    """
    Inserts or updates a file's metrics. Returns its file_id.
    Any of classes/functions/methods/async_functions/imports/lines may be
    passed; missing ones default to 0.

        set_file("flask", "src/app.py", "Python", functions=12, lines=340)
    """
    full = {k: int(metrics.get(k, 0) or 0) for k in METRIC_FIELDS}
    conn = _pg()
    try:
        file_id = _save_file(conn, repo, path, language, full)
        conn.commit()
        return file_id
    finally:
        conn.close()


# ── Business rules (PostgreSQL) ───────────────────────────────────────────────

def get_rules(repo: str, path: str = None):
    """
    All rules for a repo as {file_path: [rules]}, or a plain list of rules
    when `path` is given.
    """
    conn = _pg()
    try:
        by_file = _load_rules(conn, repo)
    finally:
        conn.close()
    return by_file.get(path, []) if path else by_file


def set_rules(repo: str, path: str, rules: list) -> int:
    """
    Replaces a file's business rules and marks it extracted. Creates the file
    row if it doesn't exist yet. Returns the number of rules stored.
    """
    existing = get_file(repo, path)
    file_id = existing["file_id"] if existing else set_file(repo, path, "Other")

    conn = _pg()
    try:
        _save_rules(conn, file_id, list(rules))
        conn.commit()
    finally:
        conn.close()
    return len(rules)


# ── Key points (PostgreSQL) ───────────────────────────────────────────────────

def get_key_points(repo: str) -> list:
    """The repo's key points, ordered by index."""
    conn = _pg()
    try:
        return _load_key_points(conn, repo)
    finally:
        conn.close()


def get_key_point(repo: str, index: int):
    """One key point by index, or None."""
    points = get_key_points(repo)
    return points[index] if 0 <= index < len(points) else None


def set_key_points(repo: str, points: list) -> int:
    """Replaces the repo's key points. Returns how many were stored."""
    conn = _pg()
    try:
        _save_key_points(conn, repo, list(points))
        conn.commit()
    finally:
        conn.close()
    return len(points)


# ── Dependencies (Kùzu) ───────────────────────────────────────────────────────

def _kuzu():
    from kb.graph.kuzu_store import get_connection
    return get_connection()


def get_dependencies(repo: str, path: str) -> list:
    """Files that `path` depends on (outgoing DEPENDS_ON)."""
    try:
        from kb.graph.kuzu_store import rows_to_dicts
        res = _kuzu().execute("""
            MATCH (f:File {repository_name: $repo, relative: $rel})-[:DEPENDS_ON]->(d:File)
            RETURN d.relative ORDER BY d.relative
        """, {"repo": repo, "rel": path})
        return [r[0] for r in rows_to_dicts(res)]
    except Exception:
        return []


def get_dependents(repo: str, path: str) -> list:
    """Files that depend on `path` (incoming DEPENDS_ON)."""
    try:
        from kb.graph.kuzu_store import rows_to_dicts
        res = _kuzu().execute("""
            MATCH (s:File {repository_name: $repo})-[:DEPENDS_ON]->(f:File {relative: $rel})
            RETURN s.relative ORDER BY s.relative
        """, {"repo": repo, "rel": path})
        return [r[0] for r in rows_to_dicts(res)]
    except Exception:
        return []


def set_dependency(repo: str, source: str, target: str) -> bool:
    """
    Adds a DEPENDS_ON edge. Both files must already exist as nodes (run the
    scan stage first). Returns True if the edge is present afterwards.
    """
    from kb.graph.kuzu_store import uid_for, scalar
    conn = _kuzu()
    conn.execute("""
        MATCH (a:File {uid: $s}), (b:File {uid: $t})
        MERGE (a)-[:DEPENDS_ON]->(b)
    """, {"s": uid_for(repo, source), "t": uid_for(repo, target)})
    return target in get_dependencies(repo, source)


def _count_edges(repo: str) -> int:
    try:
        from kb.graph.kuzu_store import scalar
        return scalar(_kuzu().execute("""
            MATCH (a:File {repository_name: $repo})-[r:DEPENDS_ON]->() RETURN count(r)
        """, {"repo": repo}))
    except Exception:
        return 0


# ── Vectors (ChromaDB) ────────────────────────────────────────────────────────

def collection_name(repo: str) -> str:
    """The repo's ChromaDB collection name."""
    safe = "".join(c if c.isalnum() else "_" for c in (repo or "").lower())
    return f"{CHROMA_COLLECTION}_{safe}"


def _collection(repo: str, create: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    name = collection_name(repo)
    if create:
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    if name not in [c.name for c in client.list_collections()]:
        return None
    return client.get_collection(name)


def get_vector(repo: str, path: str) -> list:
    """One file's key-point vector, or None."""
    try:
        col = _collection(repo)
        if col is None:
            return None
        got = col.get(ids=[path], include=["embeddings"])
        embs = got.get("embeddings")
        if embs is None or not len(embs):
            return None
        return [float(x) for x in embs[0]]   # plain floats, not numpy
    except Exception:
        return None


def get_vectors(repo: str) -> dict:
    """Every file's vector: {file_path: [scores]}."""
    try:
        col = _collection(repo)
        if col is None:
            return {}
        got = col.get(include=["embeddings"])
        return {i: [float(x) for x in e]
                for i, e in zip(got["ids"], got["embeddings"])}
    except Exception:
        return {}


def set_vector(repo: str, path: str, vector: list, rules: list = None) -> bool:
    """
    Stores a file's key-point vector. `vector` length must match the repo's
    key-point count — ChromaDB locks a collection to one dimension.
    """
    col = _collection(repo, create=True)
    key_points = get_key_points(repo)
    top = max(range(len(vector)), key=lambda k: vector[k]) if vector else 0
    import json as _json
    col.upsert(
        ids=[path],
        embeddings=[list(vector)],
        metadatas=[{
            "relative": path,
            "name": path.split("/")[-1],
            "rule_count": len(rules or []),
            "top_kp": key_points[top] if top < len(key_points) else "",
            "top_kp_index": top,
            "top_score": round(float(vector[top]), 4) if vector else 0.0,
            "kp_scores": _json.dumps({f"kp_{k}": float(v) for k, v in enumerate(vector)}),
        }],
        documents=["\n\n".join(rules or [])],
    )
    return True


def _count_vectors(repo: str) -> int:
    try:
        col = _collection(repo)
        return col.count() if col is not None else 0
    except Exception:
        return 0


# ── Convenience ───────────────────────────────────────────────────────────────

def get_everything(repo: str, path: str) -> dict:
    """
    Everything the knowledge base knows about one file, from all three stores.
    """
    return {
        "file": get_file(repo, path),
        "rules": get_rules(repo, path),
        "dependencies": get_dependencies(repo, path),
        "dependents": get_dependents(repo, path),
        "vector": get_vector(repo, path),
    }


if __name__ == "__main__":
    import sys
    repos = list_repos()
    if not repos:
        sys.exit("Knowledge base is empty.")
    print("Repositories:")
    for r in repos:
        info = get_repo(r)
        print(f"  {r}: {info.get('total_files', 0)} files, "
              f"{info.get('total_rules', 0)} rules, "
              f"{info.get('total_key_points', 0)} key points, "
              f"{info['dependencies']} deps, {info['vectors']} vectors")
        
    print("KEY POINTS")
    print(get_key_points("flask"))                        # ['Serves HTTP.', 'Manages sessions.', ...]
    print("\tSpecific Key Point:")
    print("\t", get_key_point("flask", 1))                      # 'Manages sessions.'
    print("\nBUSINESS RULES: ") 
    print(get_rules("flask", "src/app.py"))               # ['Rule one.', 'Rule two.']
    print(get_rules("flask"))                             # {file_path: [rules], ...}k 
    print("\nFILE METRICS: ")
    print(get_file("flask", "src/app.py"))                # metrics dict
    print(get_dependencies("flask", "src/app.py"))        # what it imports
    print(get_dependents("flask", "src/app.py"))
    
    print("\nVECTORS")          # what imports it
    print(get_vector("flask", "src/app.py"))              # [0.91, 0.22, 0.05]