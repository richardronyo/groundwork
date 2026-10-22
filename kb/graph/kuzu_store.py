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


def get_connection(db_path: str = None, read_only: bool = False):
    """
    Opens the Kùzu database.

    Kùzu allows EITHER one read-write connection OR several read-only ones. A
    read-write handle blocks everything else, which is why an open notebook
    kernel makes the CLI tools fail with "Could not set lock on file".

    Tools that only read (diagrams, reports, queries, the inspector) should pass
    read_only=True so they can run alongside each other and alongside an open
    notebook. Only ingestion needs read-write.
    """
    try:
        import kuzu
    except ImportError:
        raise ImportError("kuzu not installed. Run: pip install kuzu")

    path = db_path or KUZU_DB_PATH
    p = Path(path)

    if read_only and not p.exists():
        raise RuntimeError(
            f"No Kùzu database at {path}. Run the scan stage first:\n"
            f"  ./ingestion_pipeline.sh <repo> --only scan")

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
    try:
        db = kuzu.Database(str(p), read_only=read_only)
    except RuntimeError as e:
        if "lock" in str(e).lower():
            raise RuntimeError(
                f"Kùzu database at {path} is locked by another process.\n"
                f"  Kùzu allows one read-write connection at a time.\n"
                f"  • Close any open Jupyter kernel using the graph "
                f"(Kernel → Restart), or\n"
                f"  • stop any other Groundwork command still running, or\n"
                f"  • re-run this tool read-only if it only needs to read."
            ) from None
        raise
    conn = kuzu.Connection(db)
    if not read_only:
        ensure_schema(conn)        # schema changes need write access
    # Keep a reference to the Database so it isn't garbage collected
    conn._db = db
    return conn


def get_reader(db_path: str = None):
    """Read-only connection — safe to open alongside other readers."""
    return get_connection(db_path, read_only=True)


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

# ── Inspection ────────────────────────────────────────────────────────────────

def summarize(conn, repo_name: str = None) -> dict:
    """Counts of everything in the graph, optionally scoped to one repository."""
    if repo_name:
        p = {"repo": repo_name}
        return {
            "repository": repo_name,
            "files": scalar(conn.execute(
                "MATCH (f:File {repository_name: $repo}) RETURN count(f)", p)),
            "directories": scalar(conn.execute(
                "MATCH (d:Directory {repository_name: $repo}) RETURN count(d)", p)),
            "contains": scalar(conn.execute(
                "MATCH (d:Directory {repository_name: $repo})-[r:CONTAINS]->() "
                "RETURN count(r)", p)),
            "same_dir": scalar(conn.execute(
                "MATCH (a:File {repository_name: $repo})-[r:SAME_DIR]->() "
                "RETURN count(r)", p)),
            "depends_on": scalar(conn.execute(
                "MATCH (a:File {repository_name: $repo})-[r:DEPENDS_ON]->() "
                "RETURN count(r)", p)),
        }
    return {
        "repository": "(all)",
        "files": scalar(conn.execute("MATCH (f:File) RETURN count(f)")),
        "directories": scalar(conn.execute("MATCH (d:Directory) RETURN count(d)")),
        "contains": scalar(conn.execute("MATCH ()-[r:CONTAINS]->() RETURN count(r)")),
        "same_dir": scalar(conn.execute("MATCH ()-[r:SAME_DIR]->() RETURN count(r)")),
        "depends_on": scalar(conn.execute("MATCH ()-[r:DEPENDS_ON]->() RETURN count(r)")),
    }


def languages(conn, repo_name: str) -> list:
    """[(language, file_count)] for one repository."""
    return [(r[0], r[1]) for r in rows_to_dicts(conn.execute("""
        MATCH (f:File {repository_name: $repo})
        RETURN f.language, count(f) AS c ORDER BY c DESC
    """, {"repo": repo_name}))]


def sample_files(conn, repo_name: str, limit: int = 10) -> list:
    return [r[0] for r in rows_to_dicts(conn.execute("""
        MATCH (f:File {repository_name: $repo})
        RETURN f.relative ORDER BY f.relative LIMIT $n
    """, {"repo": repo_name, "n": limit}))]


def top_dependencies(conn, repo_name: str, limit: int = 10) -> list:
    """[(file, dependent_count)] — the most imported files."""
    return [(r[0], r[1]) for r in rows_to_dicts(conn.execute("""
        MATCH (s:File {repository_name: $repo})-[:DEPENDS_ON]->(t:File)
        RETURN t.relative, count(s) AS c ORDER BY c DESC LIMIT $n
    """, {"repo": repo_name, "n": limit}))]


def main():
    """Prints what is in the Kùzu graph. Run: python3 -m kb.graph.kuzu_store"""
    import argparse
    ap = argparse.ArgumentParser(description="Inspect the Kùzu graph store")
    ap.add_argument("--repo", default=None, help="Scope to one repository")
    ap.add_argument("--db", default=None, help=f"Database path (default: {KUZU_DB_PATH})")
    ap.add_argument("--query", default=None, help="Run an arbitrary Cypher query")
    ap.add_argument("--limit", type=int, default=10, help="Rows to sample (default: 10)")
    args = ap.parse_args()

    path = args.db or KUZU_DB_PATH
    if not Path(path).exists():
        print(f"\n  No Kùzu database at {path}.")
        print(f"  Run the scan stage first: ./ingestion_pipeline.sh <repo> --only scan\n")
        return

    try:
        conn = get_reader(path)          # read-only: coexists with a notebook
    except RuntimeError as e:
        print(f"\n  {e}\n")
        return
    print(f"\n  Kùzu database: {path}  (read-only)")

    if args.query:
        res = conn.execute(args.query)
        rows = rows_to_dicts(res)
        print(f"  {len(rows)} row(s)\n")
        for r in rows[:args.limit]:
            print("   ", r)
        if len(rows) > args.limit:
            print(f"    … and {len(rows) - args.limit} more")
        print()
        return

    repos = list_repos(conn)
    print(f"  Repositories : {', '.join(repos) if repos else '(none)'}")

    overall = summarize(conn)
    print(f"\n  ── Whole graph ──")
    print(f"    Files        : {overall['files']:,}")
    print(f"    Directories  : {overall['directories']:,}")
    print(f"    CONTAINS     : {overall['contains']:,}")
    print(f"    SAME_DIR     : {overall['same_dir']:,}")
    print(f"    DEPENDS_ON   : {overall['depends_on']:,}")

    targets = [args.repo] if args.repo else repos
    for repo in targets:
        if not repo:
            continue
        s = summarize(conn, repo)
        print(f"\n  ── {repo} ──")
        print(f"    Files        : {s['files']:,}")
        print(f"    Directories  : {s['directories']:,}")
        print(f"    CONTAINS     : {s['contains']:,}")
        print(f"    SAME_DIR     : {s['same_dir']:,}")
        print(f"    DEPENDS_ON   : {s['depends_on']:,}")

        langs = languages(conn, repo)
        if langs:
            print(f"    Languages    : " +
                  ", ".join(f"{l} ({n})" for l, n in langs[:6]))

        top = top_dependencies(conn, repo, 5)
        if top:
            print(f"    Most imported:")
            for f, n in top:
                print(f"       {n:>4}x  {f}")
        elif s["files"]:
            print(f"    Most imported: (no DEPENDS_ON edges — run the deps stage)")

        files = sample_files(conn, repo, 5)
        if files:
            print(f"    Sample files :")
            for f in files:
                print(f"       {f}")
    print()


if __name__ == "__main__":
    main()
