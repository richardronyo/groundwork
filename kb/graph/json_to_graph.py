"""
Groundwork — Neo4j File Structure Ingester
Reads the JSON output from tree_to_json.py and populates a Neo4j graph
with File nodes and CONTAINS / SAME_DIR relationship edges.

Usage:
    python3 json_to_graph.py <files.json>
    python3 json_to_graph.py <files.json> --clear

"""

import json
import argparse
import os
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv


# ── Neo4j connection ──────────────────────────────────────────────────────────
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── Scripting / code file filter ──────────────────────────────────────────────

CODE_LANGUAGES = {
    "Python",
    "JavaScript",
    "TypeScript",
    "JavaScript (React)",
    "TypeScript (React)",
    "Java",
    "Kotlin",
    "C#",
    "C++",
    "C",
    "C/C++ Header",
    "Go",
    "Rust",
    "Ruby",
    "PHP",
    "Swift",
    "Shell",
    "Batch",
    "SQL",
}

# ── Schema ────────────────────────────────────────────────────────────────────
#
#  (:Directory {relative, name, depth})
#  (:File      {relative, name, extension, language, size_bytes})
#
#  (:Directory)-[:CONTAINS]->(:File)
#  (:Directory)-[:CONTAINS]->(:Directory)
#  (:File)-[:SAME_DIR]->(:File)
#
# ─────────────────────────────────────────────────────────────────────────────


class GraphIngester:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"  Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def create_constraints(self):
        with self.driver.session() as s:
            s.run("""
                CREATE CONSTRAINT file_relative_unique IF NOT EXISTS
                FOR (f:File) REQUIRE f.relative IS UNIQUE
            """)
            s.run("""
                CREATE CONSTRAINT dir_relative_unique IF NOT EXISTS
                FOR (d:Directory) REQUIRE d.relative IS UNIQUE
            """)
        print("  Constraints ensured.")

    def clear_graph(self):
        with self.driver.session() as s:
            s.run("MATCH (n:File) DETACH DELETE n")
            s.run("MATCH (n:Directory) DETACH DELETE n")
        print("  Cleared existing File and Directory nodes.")

    # ── Directory helpers ─────────────────────────────────────────────────────

    def _ensure_directory_chain(self, tx, relative: str):
        parts = Path(relative).parent.parts

        if not parts:
            tx.run("""
                MERGE (d:Directory {relative: ""})
                SET d.name = "(root)", d.depth = 0
            """)
            return ""

        dir_relatives = []
        for i in range(len(parts)):
            dir_relatives.append("/".join(parts[: i + 1]))

        tx.run("""
            MERGE (d:Directory {relative: ""})
            SET d.name = "(root)", d.depth = 0
        """)

        for i, rel in enumerate(dir_relatives):
            name = parts[i]
            tx.run("""
                MERGE (d:Directory {relative: $relative})
                SET d.name  = $name,
                    d.depth = $depth
            """, relative=rel, name=name, depth=i + 1)

        all_dirs = [""] + dir_relatives
        for i in range(len(all_dirs) - 1):
            tx.run("""
                MATCH (parent:Directory {relative: $parent})
                MATCH (child:Directory  {relative: $child})
                MERGE (parent)-[:CONTAINS]->(child)
            """, parent=all_dirs[i], child=all_dirs[i + 1])

        return dir_relatives[-1]

    # ── File node ─────────────────────────────────────────────────────────────

    def _ingest_file(self, tx, file_meta: dict):
        relative = file_meta["relative"]
        parent_relative = self._ensure_directory_chain(tx, relative)

        tx.run("""
            MERGE (f:File {relative: $relative})
            SET f.name       = $name,
                f.extension  = $extension,
                f.language   = $language,
                f.size_bytes = $size_bytes
        """,
            relative   = relative,
            name       = file_meta["name"],
            extension  = file_meta.get("extension", ""),
            language   = file_meta.get("language", "Other"),
            size_bytes = file_meta.get("size_bytes", 0),
        )

        tx.run("""
            MATCH (d:Directory {relative: $dir_rel})
            MATCH (f:File      {relative: $file_rel})
            MERGE (d)-[:CONTAINS]->(f)
        """, dir_rel=parent_relative, file_rel=relative)

    # ── SAME_DIR edges ────────────────────────────────────────────────────────

    def _create_same_dir_edges(self, tx):
        tx.run("""
            MATCH (d:Directory)-[:CONTAINS]->(a:File)
            MATCH (d)-[:CONTAINS]->(b:File)
            WHERE a.relative < b.relative
            MERGE (a)-[:SAME_DIR]->(b)
        """)

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest(self, files: list[dict]):
        total = len(files)
        print(f"\n  Ingesting {total} code files into Neo4j...\n")

        with self.driver.session() as session:
            for i, file_meta in enumerate(files):
                session.execute_write(self._ingest_file, file_meta)
                pct = int((i + 1) / total * 40)
                bar = "█" * pct + "░" * (40 - pct)
                print(f"\r  [{bar}] {i+1}/{total}", end="", flush=True)

        print(f"\n\n  Creating SAME_DIR edges...")
        with self.driver.session() as session:
            session.execute_write(self._create_same_dir_edges)

        print("  Done.\n")
        self._print_summary()

    def _print_summary(self):
        with self.driver.session() as session:
            file_count = session.run("MATCH (f:File) RETURN count(f) AS c").single()["c"]
            dir_count  = session.run("MATCH (d:Directory) RETURN count(d) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            lang_rows  = session.run("""
                MATCH (f:File)
                RETURN f.language AS lang, count(f) AS c
                ORDER BY c DESC LIMIT 10
            """).data()

        print("  ┌─ Graph Summary ──────────────────────────────┐")
        print(f"  │  File nodes      : {file_count:<6}                    │")
        print(f"  │  Directory nodes : {dir_count:<6}                    │")
        print(f"  │  Total edges     : {edge_count:<6}                    │")
        print(f"  │                                              │")
        print(f"  │  Files by language:                          │")
        for row in lang_rows:
            bar = "█" * min(row["c"] // 10 + 1, 20)
            print(f"  │    {row['lang']:<22} {bar:<20} {row['c']} │")
        print("  └──────────────────────────────────────────────┘\n")
        print("  Neo4j browser → https://console.neo4j.io")
        print("  Run: MATCH (n) RETURN n\n")


# ── Reference queries ─────────────────────────────────────────────────────────

REFERENCE_QUERIES = """
──────────────────────────────────────────────────────────
  Useful Cypher queries for your graph:

  // Full repo tree
  MATCH (n) RETURN n

  // All files inside a specific directory
  MATCH (d:Directory {name: "src"})-[:CONTAINS]->(f:File)
  RETURN f.name, f.language

  // All C# files
  MATCH (f:File {language: "C#"})
  RETURN f.relative ORDER BY f.relative

  // Directory tree only
  MATCH p=(root:Directory {relative: ""})-[:CONTAINS*]->(d:Directory)
  RETURN p

  // Files grouped by language
  MATCH (f:File)
  RETURN f.language, count(f) AS total
  ORDER BY total DESC

  // Files that share a directory with a given file
  MATCH (f:File {name: "app.py"})-[:SAME_DIR]-(neighbour:File)
  RETURN neighbour.name, neighbour.language
──────────────────────────────────────────────────────────
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — ingest tree_to_json.py output into Neo4j"
    )
    parser.add_argument("json_file", help="Path to JSON file list from tree_to_json.py")
    parser.add_argument("--clear",   action="store_true", help="Clear existing nodes before ingesting")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: '{json_path}' not found.")
        sys.exit(1)

    with open(json_path) as f:
        all_files = json.load(f)

    # ── Filter to code files only ─────────────────────────────────────────────
    files = [f for f in all_files if f.get("language") in CODE_LANGUAGES]
    skipped = len(all_files) - len(files)
    print(f"\n  Loaded {len(all_files)} files from {json_path}")
    print(f"  Keeping {len(files)} code files, skipping {skipped} non-code files")
    print(f"  Languages: {', '.join(sorted({f['language'] for f in files}))}")

    ingester = GraphIngester(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    try:
        ingester.create_constraints()
        if args.clear:
            ingester.clear_graph()
        ingester.ingest(files)
        print(REFERENCE_QUERIES)
    finally:
        ingester.close()


if __name__ == "__main__":
    main()