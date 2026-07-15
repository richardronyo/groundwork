"""
Groundwork — Kùzu File Structure Ingester
Reads the JSON output from tree_to_json.py and populates the embedded Kùzu
graph with File / Directory nodes and CONTAINS / SAME_DIR edges.

Replaces the Neo4j version: no server, no cloud, no per-node network calls.
All rows are built in memory and written in UNWIND batches.

Usage:
    python3 -m kb.graph.json_to_graph <files.json> --repo flask
    python3 -m kb.graph.json_to_graph <files.json> --repo flask --clear

Requirements:
    pip install kuzu
"""

import json
import argparse
import sys
from pathlib import Path

from kb.graph.kuzu_store import (
    get_connection, uid_for, batched, scalar, rows_to_dicts,
    clear_repo, KUZU_DB_PATH,
)

# ── Scripting / code file filter ──────────────────────────────────────────────

CODE_LANGUAGES = {
    "Python", "JavaScript", "TypeScript",
    "JavaScript (React)", "TypeScript (React)",
    "Java", "Kotlin", "C#", "C++", "C", "C/C++ Header",
    "Go", "Rust", "Ruby", "PHP", "Swift", "Shell", "Batch", "SQL",
}


class GraphIngester:

    def __init__(self, repo_name: str, db_path: str = None):
        self.repo_name = repo_name
        self.db_path = db_path or KUZU_DB_PATH
        self.conn = get_connection(self.db_path)
        print(f"  Kùzu graph: {self.db_path} (embedded — no server)")
        print(f"  Repository: {repo_name}")

    def close(self):
        pass  # embedded; nothing to disconnect

    def _uid(self, relative: str) -> str:
        return uid_for(self.repo_name, relative)

    def clear_graph(self):
        removed = clear_repo(self.conn, self.repo_name)
        print(f"  Cleared {removed} existing nodes for repo '{self.repo_name}'.")

    # ── Row building (pure Python, no DB calls) ───────────────────────────────

    def _build_rows(self, files: list[dict]):
        """
        Walks the file list once and produces every row we need:
          dir_rows, file_rows, contains_dir_dir, contains_dir_file
        Building these in memory means the DB is written in a handful of
        batched statements instead of thousands of individual queries.
        """
        repo = self.repo_name
        dirs = {}                 # relative -> row
        file_rows = []
        contains_dd = set()       # (parent_rel, child_rel)
        contains_df = []          # (dir_rel, file_rel)

        # Root directory always exists
        dirs[""] = {"uid": self._uid(""), "relative": "", "name": "(root)",
                    "depth": 0, "repo": repo}

        for fm in files:
            relative = fm["relative"]
            parts = Path(relative).parent.parts

            # Ensure every ancestor directory exists, and link the chain
            chain = [""]
            for i in range(len(parts)):
                rel = "/".join(parts[: i + 1])
                chain.append(rel)
                if rel not in dirs:
                    dirs[rel] = {"uid": self._uid(rel), "relative": rel,
                                 "name": parts[i], "depth": i + 1, "repo": repo}
            for i in range(len(chain) - 1):
                contains_dd.add((chain[i], chain[i + 1]))

            parent_rel = chain[-1]
            file_rows.append({
                "uid": self._uid(relative),
                "relative": relative,
                "name": fm["name"],
                "extension": fm.get("extension", ""),
                "language": fm.get("language", "Other"),
                "size_bytes": int(fm.get("size_bytes", 0) or 0),
                "repo": repo,
            })
            contains_df.append((parent_rel, relative))

        return (list(dirs.values()), file_rows,
                sorted(contains_dd), contains_df)

    # ── Batched writes ────────────────────────────────────────────────────────

    def _insert_dirs(self, dir_rows):
        for chunk in batched(dir_rows):
            self.conn.execute("""
                UNWIND $rows AS row
                MERGE (d:Directory {uid: row.uid})
                SET d.relative = row.relative,
                    d.name = row.name,
                    d.depth = row.depth,
                    d.repository_name = row.repo
            """, {"rows": chunk})

    def _insert_files(self, file_rows):
        total = len(file_rows)
        done = 0
        for chunk in batched(file_rows):
            self.conn.execute("""
                UNWIND $rows AS row
                MERGE (f:File {uid: row.uid})
                SET f.relative = row.relative,
                    f.name = row.name,
                    f.extension = row.extension,
                    f.language = row.language,
                    f.size_bytes = row.size_bytes,
                    f.repository_name = row.repo
            """, {"rows": chunk})
            done += len(chunk)
            pct = int(done / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {done}/{total}", end="", flush=True)
        print()

    def _insert_contains(self, contains_dd, contains_df):
        # Directory -> Directory
        for chunk in batched([{"s": self._uid(a), "t": self._uid(b)}
                              for a, b in contains_dd]):
            self.conn.execute("""
                UNWIND $rows AS row
                MATCH (a:Directory {uid: row.s}), (b:Directory {uid: row.t})
                MERGE (a)-[:CONTAINS]->(b)
            """, {"rows": chunk})

        # Directory -> File
        for chunk in batched([{"s": self._uid(a), "t": self._uid(b)}
                              for a, b in contains_df]):
            self.conn.execute("""
                UNWIND $rows AS row
                MATCH (a:Directory {uid: row.s}), (b:File {uid: row.t})
                MERGE (a)-[:CONTAINS]->(b)
            """, {"rows": chunk})

    def _create_same_dir_edges(self):
        self.conn.execute("""
            MATCH (d:Directory {repository_name: $repo})-[:CONTAINS]->(a:File)
            MATCH (d)-[:CONTAINS]->(b:File)
            WHERE a.relative < b.relative
            MERGE (a)-[:SAME_DIR]->(b)
        """, {"repo": self.repo_name})

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, files: list[dict]):
        total = len(files)
        print(f"\n  Building rows for {total} code files...")
        dir_rows, file_rows, contains_dd, contains_df = self._build_rows(files)
        print(f"  {len(dir_rows)} directories, {len(file_rows)} files, "
              f"{len(contains_dd) + len(contains_df)} CONTAINS edges")

        print(f"\n  Writing directories...")
        self._insert_dirs(dir_rows)

        print(f"  Writing files...")
        self._insert_files(file_rows)

        print(f"  Writing CONTAINS edges...")
        self._insert_contains(contains_dd, contains_df)

        print(f"  Creating SAME_DIR edges...")
        self._create_same_dir_edges()

        print("  Done.\n")
        self._print_summary()

    def _print_summary(self):
        repo = self.repo_name
        c = self.conn
        file_count = scalar(c.execute(
            "MATCH (f:File {repository_name: $repo}) RETURN count(f)", {"repo": repo}))
        dir_count = scalar(c.execute(
            "MATCH (d:Directory {repository_name: $repo}) RETURN count(d)", {"repo": repo}))
        contains = scalar(c.execute(
            "MATCH (a:Directory {repository_name: $repo})-[r:CONTAINS]->() RETURN count(r)",
            {"repo": repo}))
        same_dir = scalar(c.execute(
            "MATCH (a:File {repository_name: $repo})-[r:SAME_DIR]->() RETURN count(r)",
            {"repo": repo}))
        depends = scalar(c.execute(
            "MATCH (a:File {repository_name: $repo})-[r:DEPENDS_ON]->() RETURN count(r)",
            {"repo": repo}))
        lang_rows = rows_to_dicts(c.execute("""
            MATCH (f:File {repository_name: $repo})
            RETURN f.language, count(f) AS c
            ORDER BY c DESC LIMIT 10
        """, {"repo": repo}))

        print("  ┌─ Graph Summary ──────────────────────────────┐")
        print(f"  │  File nodes      : {file_count:<6}                    │")
        print(f"  │  Directory nodes : {dir_count:<6}                    │")
        print(f"  │  CONTAINS edges  : {contains:<6}                    │")
        print(f"  │  SAME_DIR edges  : {same_dir:<6}                    │")
        print(f"  │  DEPENDS_ON edges: {depends:<6}                    │")
        print(f"  │                                              │")
        print(f"  │  Files by language:                          │")
        for lang, count in lang_rows:
            bar = "█" * min(count // 10 + 1, 20)
            print(f"  │    {str(lang):<22} {bar:<20} {count} │")
        print("  └──────────────────────────────────────────────┘\n")


# ── Reference queries ─────────────────────────────────────────────────────────

REFERENCE_QUERIES = """
──────────────────────────────────────────────────────────
  Useful Cypher queries (run via kuzu.Connection.execute):

  // All files inside a specific directory
  MATCH (d:Directory {name: "src"})-[:CONTAINS]->(f:File)
  RETURN f.name, f.language

  // All C# files in a repo
  MATCH (f:File {language: "C#", repository_name: "nopCommerce"})
  RETURN f.relative ORDER BY f.relative

  // Direct dependencies of a file
  MATCH (f:File {relative: "app.py"})-[:DEPENDS_ON]->(d:File)
  RETURN d.relative

  // Transitive dependencies (variable-length path)
  MATCH (f:File {relative: "app.py"})-[:DEPENDS_ON*1..3]->(d:File)
  RETURN DISTINCT d.relative

  // Files grouped by language
  MATCH (f:File) RETURN f.language, count(f) AS total ORDER BY total DESC

  // Files that share a directory
  MATCH (f:File {name: "app.py"})-[:SAME_DIR]-(n:File)
  RETURN n.name, n.language
──────────────────────────────────────────────────────────
"""


def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — ingest tree_to_json.py output into Kùzu"
    )
    parser.add_argument("json_file", help="Path to JSON file list from tree_to_json.py")
    parser.add_argument("--repo", required=True, help="Repository name (scopes all nodes)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear THIS repo's nodes before ingesting")
    parser.add_argument("--db", default=None,
                        help=f"Kùzu database path (default: {KUZU_DB_PATH})")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: '{json_path}' not found.")
        sys.exit(1)

    with open(json_path) as f:
        all_files = json.load(f)

    files = [f for f in all_files if f.get("language") in CODE_LANGUAGES]
    skipped = len(all_files) - len(files)
    print(f"\n  Loaded {len(all_files)} files from {json_path}")
    print(f"  Keeping {len(files)} code files, skipping {skipped} non-code files")
    if files:
        print(f"  Languages: {', '.join(sorted({f['language'] for f in files}))}")

    ingester = GraphIngester(args.repo, args.db)
    try:
        if args.clear:
            ingester.clear_graph()
        ingester.ingest(files)
        print(REFERENCE_QUERIES)
    finally:
        ingester.close()


if __name__ == "__main__":
    main()