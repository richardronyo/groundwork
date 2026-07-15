"""
Groundwork — Kùzu Graph Store (embedded, replaces Neo4j)

Kùzu is an embedded graph database: no server, no cloud account, no network
round-trips. The whole graph lives in a single local file (./kuzu_db), the way
SQLite does — do NOT mkdir this path, Kùzu creates the file itself.

Unlike Neo4j, Kùzu is SCHEMA-FIRST — node and relationship tables must be
declared up front. This module owns that schema and the connection so every
other module shares one definition.

Schema
------
  (:Directory {uid, relative, name, depth, repository_name})
  (:File      {uid, relative, name, extension, language, size_bytes,
               repository_name})

  (:Directory)-[:CONTAINS]->(:File)
  (:Directory)-[:CONTAINS]->(:Directory)
  (:File)-[:SAME_DIR]->(:File)
  (:File)-[:DEPENDS_ON]->(:File)

uid = "<repo>::<relative>" so the same path in different repositories maps to
distinct nodes (multi-repo scoping, same scheme as the Neo4j version).

Requirements:
    pip install kuzu
"""

import os
from pathlib import Path

KUZU_DB_PATH = os.getenv("KUZU_DB_PATH", "./kuzu_db")

# Rows per UNWIND batch. Batching is what keeps ingestion fast — writing one
# row per query is orders of magnitude slower.
BATCH_SIZE = 1000


def uid_for(repo_name: str, relative: str) -> str:
    """Synthetic node key: scopes a path to its repository."""
    return f"{repo_name}::{relative}"


def get_connection(db_path: str = None):
    """Opens (creating if needed) the Kùzu database and ensures the schema."""
    try:
        import kuzu
    except ImportError:
        raise ImportError("kuzu not installed. Run: pip install kuzu")

    path = db_path or KUZU_DB_PATH
    p = Path(path)

    # Kùzu stores the whole database in a SINGLE FILE, not a directory. If an
    # empty directory is sitting at the path (e.g. created by a stray mkdir),
    # Kùzu refuses to open it — remove it so the DB can be created.
    if p.is_dir():
        if any(p.iterdir()):
            raise RuntimeError(
                f"Kùzu database path '{path}' is a non-empty directory. "
                f"Kùzu needs a file path here. Move or delete that directory, "
                f"or set KUZU_DB_PATH to a different location."
            )
        p.rmdir()

    p.parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(p))
    conn = kuzu.Connection(db)
    ensure_schema(conn)
    # Keep a reference to the Database so it isn't garbage collected
    conn._db = db
    return conn


def ensure_schema(conn):
    """
    Creates node/rel tables if absent. Kùzu requires an explicit schema, so
    this is the equivalent of Neo4j's constraint setup — and the PRIMARY KEY
    on uid gives us the uniqueness the Neo4j version enforced by constraint.
    """
    statements = [
        """CREATE NODE TABLE IF NOT EXISTS Directory(
               uid STRING,
               relative STRING,
               name STRING,
               depth INT64,
               repository_name STRING,
               PRIMARY KEY(uid))""",
        """CREATE NODE TABLE IF NOT EXISTS File(
               uid STRING,
               relative STRING,
               name STRING,
               extension STRING,
               language STRING,
               size_bytes INT64,
               repository_name STRING,
               PRIMARY KEY(uid))""",
        """CREATE REL TABLE IF NOT EXISTS CONTAINS(
               FROM Directory TO File,
               FROM Directory TO Directory)""",
        "CREATE REL TABLE IF NOT EXISTS SAME_DIR(FROM File TO File)",
        "CREATE REL TABLE IF NOT EXISTS DEPENDS_ON(FROM File TO File)",
    ]
    for stmt in statements:
        conn.execute(stmt)


def batched(rows, size: int = BATCH_SIZE):
    """Yields successive chunks of `rows` for UNWIND batch writes."""
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def rows_to_dicts(result):
    """Drains a Kùzu QueryResult into a list of row lists."""
    out = []
    while result.has_next():
        out.append(result.get_next())
    return out


def scalar(result, default=0):
    """Reads a single scalar (e.g. a count) out of a QueryResult."""
    if result.has_next():
        return result.get_next()[0]
    return default


# ── Repo-scoped helpers shared by the pipeline ────────────────────────────────

def clear_repo(conn, repo_name: str) -> int:
    """Deletes one repository's nodes (and their edges). Returns nodes removed."""
    n = scalar(conn.execute(
        "MATCH (f:File {repository_name: $repo}) RETURN count(f)",
        {"repo": repo_name}))
    n += scalar(conn.execute(
        "MATCH (d:Directory {repository_name: $repo}) RETURN count(d)",
        {"repo": repo_name}))
    conn.execute("MATCH (f:File {repository_name: $repo}) DETACH DELETE f",
                 {"repo": repo_name})
    conn.execute("MATCH (d:Directory {repository_name: $repo}) DETACH DELETE d",
                 {"repo": repo_name})
    return n


def clear_all(conn) -> int:
    """Deletes every File/Directory node in the graph. Returns nodes removed."""
    n = scalar(conn.execute("MATCH (f:File) RETURN count(f)"))
    n += scalar(conn.execute("MATCH (d:Directory) RETURN count(d)"))
    conn.execute("MATCH (f:File) DETACH DELETE f")
    conn.execute("MATCH (d:Directory) DETACH DELETE d")
    return n


def list_repos(conn) -> list:
    """Repositories present in the graph."""
    res = conn.execute(
        "MATCH (f:File) RETURN DISTINCT f.repository_name ORDER BY f.repository_name")
    return [r[0] for r in rows_to_dicts(res)]


def get_dependencies(conn, file_paths: list, repo_name: str) -> dict:
    """
    { source_relative: [dependency_relative, ...] } for the given files,
    scoped to one repository. Single query, no per-file round-trips.
    """
    if not file_paths:
        return {}
    res = conn.execute("""
        MATCH (f:File {repository_name: $repo})-[:DEPENDS_ON]->(d:File)
        WHERE list_contains($paths, f.relative)
        RETURN f.relative, d.relative
    """, {"repo": repo_name, "paths": file_paths})

    out = {}
    for src, dep in rows_to_dicts(res):
        out.setdefault(src, []).append(dep)
    return out