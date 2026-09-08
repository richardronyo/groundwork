#!/usr/bin/env python3
"""
Groundwork — Repository Ingestion (parser → PG → Kùzu) without temp files.

Usage:
    python3 ingest_repo.py /path/to/repo
    python3 ingest_repo.py /path/to/repo --only graph
    python3 ingest_repo.py /path/to/repo --from deps
"""

import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to PYTHONPATH if needed
sys.path.insert(0, str(Path(__file__).parent))

# Our modules
from new_kb.graph.tree_to_json import parse_tree
from new_kb.graph.json_to_graph import GraphIngester
from new_kb.graph.file_dependencies import extract_dependencies_from_db, push_edges
from new_kb.relationaldb.initialize_db import get_connection, init_db, save_file
from new_kb.parser.parser import get_parser, get_tree, extract_file_metrics, reshape_for_db
from new_kb.graph.kuzu_store import get_connection as get_kuzu_connection
from new_kb.parser.parser_dicts import EXTENSION_TO_MODULE


def get_file_list(repo_root: Path) -> list[dict]:
    """Run tree -fi and parse output to a list of file dicts."""
    proc = subprocess.run(
        ["tree", "-fi", str(repo_root)],
        capture_output=True,
        text=True,
        check=True
    )
    lines = proc.stdout.splitlines()
    # parse_tree expects the first line as the repo root, but we already have it.
    # We'll pass the lines and the root.
    return parse_tree(lines, str(repo_root), repo_root.name)


def store_metrics(repo_root: Path, files: list[dict]):
    """
    For each file, run Tree‑sitter parser and store metrics + imports in PostgreSQL.
    Only processes files with extensions listed in EXTENSION_TO_MODULE.
    """

    conn = get_connection()
    repo_name = repo_root.name
    supported_exts = set(EXTENSION_TO_MODULE.keys())

    # Filter files to only those with supported extensions
    code_files = [f for f in files if f.get("extension", "") and f".{f['extension']}" in supported_exts]
    print(f"  Processing {len(code_files)} code files out of {len(files)} total.")

    for file_meta in code_files:
        rel_path = file_meta["relative"]   # e.g., "flask/app.py"
        full_path = repo_root / rel_path   # works because relative includes repo name
        ext = file_meta.get("extension", "")
        if not ext:
            continue

        if not full_path.exists():
            print(f"  Skipping {rel_path} – file not found")
            continue

        # Parse with Tree‑sitter
        try:
            parser = get_parser(str(full_path))
            tree = get_tree(str(full_path), parser)
            source_bytes = full_path.read_bytes()
            parser_metrics = extract_file_metrics(tree.root_node, f".{ext}", source_bytes)
        except Exception as e:
            print(f"  ⚠️ Skipping {rel_path} – {e}")
            continue

        db_metrics = reshape_for_db(parser_metrics)
        import_names = parser_metrics.get("imports", [])

        # Store in PostgreSQL
        file_id = save_file(
            conn,
            repo_name,
            rel_path,
            file_meta["language"],
            db_metrics,
            import_names=import_names
        )
        print(f"  Stored {rel_path} (id={file_id}) – "
              f"{db_metrics['functions']} funcs, "
              f"{db_metrics['classes']} classes, "
              f"{len(import_names)} imports")

    conn.close()

def build_graph(repo_name: str, files: list[dict], db_path: str = None):
    """Create File/Directory nodes and CONTAINS/SAME_DIR edges in Kùzu."""
    ingester = GraphIngester(repo_name, db_path)
    ingester.clear_graph()   # clear existing nodes for this repo
    ingester.ingest(files)   # now accepts a list
    ingester.close()


def build_dependencies(repo_name: str, repo_root: Path, files: list[dict], db_path: str = None):
    """Resolve imports from PostgreSQL and create DEPENDS_ON edges."""
    edges = extract_dependencies_from_db(files, repo_root, repo_name)
    print(f"  Resolved {len(edges)} dependencies.")
    if edges:
        kuzu_conn = get_kuzu_connection(db_path)
        push_edges(edges, kuzu_conn, repo_name)


def main():
    parser = argparse.ArgumentParser(description="Ingest a repository")
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("--from", dest="from_stage", default="init",
                        choices=["init", "parse", "graph", "deps"],
                        help="Start from this stage")
    parser.add_argument("--only", dest="only_stage", default=None,
                        choices=["init", "parse", "graph", "deps"],
                        help="Run only this stage")
    parser.add_argument("--db", default=None, help="Kùzu database path")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"Error: {repo_path} not found.")
        sys.exit(1)

    repo_name = repo_path.name

    # Initialize DB (if stage includes init)
    if args.only_stage == "init" or (args.only_stage is None and args.from_stage == "init"):
        init_db()
        if args.only_stage:
            return

    # Get file list once (needed for parse, graph, deps)
    files = get_file_list(repo_path)
    print(f"  Found {len(files)} files.")

    # Determine which stages to run
    stages = ["parse", "graph", "deps"]
    start_idx = 0 if args.from_stage == "init" else stages.index(args.from_stage) if args.from_stage in stages else 0
    end_idx = len(stages) - 1 if args.only_stage is None else stages.index(args.only_stage)

    for stage in stages[start_idx:end_idx+1]:
        print(f"\n--- Running stage: {stage} ---")
        if stage == "parse":
            store_metrics(repo_path, files)
        elif stage == "graph":
            build_graph(repo_name, files, args.db)
        elif stage == "deps":
            build_dependencies(repo_name, repo_path, files, args.db)

    print("\n✅ Ingestion complete.")


if __name__ == "__main__":
    main()