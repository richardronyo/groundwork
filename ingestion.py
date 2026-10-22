#!/usr/bin/env python3
"""
Groundwork — Repository Ingestion (parser → PG → Kùzu → LLM extraction & synthesis → ChromaDB)

Usage:
    # Full pipeline
    python3 ingest_repo.py /path/to/repo --provider openai --model gpt-4o

    # Only up to extraction (no synthesis, no graph, no embeddings)
    python3 ingest_repo.py /path/to/repo --only extract --provider openai --model gpt-4o

    # Only synthesize (rules already exist)
    python3 ingest_repo.py /path/to/repo --only synthesize --provider openai --model gpt-4o

    # Only embeddings (rules and key points already exist)
    python3 ingest_repo.py /path/to/repo --only embeddings

    # Skip LLM steps and embeddings
    python3 ingest_repo.py /path/to/repo --no-extract --no-synthesize --no-embeddings

    # Purge first
    python3 ingest_repo.py /path/to/repo --purge --from parse --provider openai --model gpt-4o
"""

import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kb.graph.tree_to_json import parse_tree
from kb.graph.json_to_graph import GraphIngester
from kb.graph.file_dependencies import extract_dependencies_from_db, push_edges
from kb.relationaldb.initialize_db import get_connection, init_db, save_file, clear_repo
from kb.parser.parser import get_parser, get_tree, extract_file_metrics, reshape_for_db
from kb.graph.kuzu_store import get_connection as get_kuzu_connection
from kb.parser.parser_dicts import EXTENSION_TO_MODULE

# New imports for LLM stages
from kb.vector.extract_business_rules import run_extraction
from kb.vector.synthesize import run_synthesis
from kb.vector.embeddings import run_embeddings


def get_file_list(repo_root: Path) -> list[dict]:
    proc = subprocess.run(
        ["tree", "-fi", str(repo_root)],
        capture_output=True,
        text=True,
        check=True
    )
    lines = proc.stdout.splitlines()
    return parse_tree(lines, str(repo_root), repo_root.name)


def store_detailed_metrics(repo_root: Path, files: list[dict]):
    """
    Parse with Tree‑sitter and insert detailed class/function data
    into the new `classes`, `class_attributes`, and `functions` tables.
    """
    conn = get_connection()
    repo_name = repo_root.name
    supported_exts = set(EXTENSION_TO_MODULE.keys())

    code_files = [f for f in files if f.get("extension", "") and f".{f['extension']}" in supported_exts]
    print(f"  Processing {len(code_files)} code files out of {len(files)} total.")

    # tree_to_json.py always prefixes `relative` with the repo name itself
    # (e.g. "flask/src/flask/app.py"), so joining it straight onto repo_root
    # (which IS the "flask" directory) doubles the repo name and every path
    # fails to exist. Strip that prefix before joining.
    name_prefix = f"{repo_name}/"

    for file_meta in code_files:
        rel_path = file_meta["relative"]
        rel_within_repo = rel_path[len(name_prefix):] if rel_path.startswith(name_prefix) else rel_path
        full_path = repo_root / rel_within_repo
        ext = file_meta.get("extension", "")
        if not ext:
            continue
        if not full_path.exists():
            print(f"  Skipping {rel_path} – file not found")
            continue

        try:
            parser = get_parser(str(full_path))
            tree = get_tree(str(full_path), parser)
            source_bytes = full_path.read_bytes()
            parser_metrics = extract_file_metrics(tree.root_node, f".{ext}", source_bytes)
        except Exception as e:
            print(f"  ⚠️ Skipping {rel_path} – {e}")
            continue

        db_metrics = reshape_for_db(parser_metrics)
        file_id = save_file(
            conn,
            repo_name,
            rel_path,
            file_meta["language"],
            db_metrics,
            import_names=parser_metrics.get("imports", [])
        )
        print(f"  Stored {rel_path} (id={file_id}) – "
              f"{db_metrics['functions']} funcs, "
              f"{db_metrics['classes']} classes, "
              f"{len(parser_metrics.get('imports', []))} imports")

        # Insert classes
        class_name_to_id = {}
        for cls in db_metrics["class_details"]:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO classes (file_id, name, bases)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (file_id, cls["name"], cls["bases"]))
                class_id = cur.fetchone()[0]
                class_name_to_id[cls["name"]] = class_id

                if cls["attributes"]:
                    attr_rows = [
                        (class_id, attr["name"], attr.get("type"), attr.get("visibility"))
                        for attr in cls["attributes"]
                    ]
                    # execute_values() is psycopg2-only (it reaches for
                    # cur.connection.encoding, which doesn't exist on a
                    # psycopg (v3) connection); psycopg3's executemany()
                    # is the portable equivalent here.
                    cur.executemany(
                        """
                        INSERT INTO class_attributes (class_id, name, type, visibility)
                        VALUES (%s, %s, %s, %s)
                        """,
                        attr_rows
                    )
            conn.commit()

        # Insert functions
        func_rows = []
        for fn in db_metrics["function_details"]:
            class_id = None
            if "containing_class" in fn and fn["containing_class"] in class_name_to_id:
                class_id = class_name_to_id[fn["containing_class"]]
            func_rows.append((
                file_id,
                class_id,
                fn["name"],
                fn["params"],
                fn["return_type"],
                fn["is_async"],
                fn["visibility"]
            ))
        if func_rows:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO functions
                        (file_id, class_id, name, params, return_type, is_async, visibility)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    func_rows
                )
                conn.commit()

    conn.close()


def build_graph(repo_name: str, files: list[dict], db_path: str = None):
    ingester = GraphIngester(repo_name, db_path)
    ingester.clear_graph()
    ingester.ingest(files)
    ingester.close()


def build_dependencies(repo_name: str, repo_root: Path, files: list[dict], db_path: str = None):
    edges = extract_dependencies_from_db(files, repo_root, repo_name)
    print(f"  Resolved {len(edges)} dependencies.")
    if edges:
        kuzu_conn = get_kuzu_connection(db_path)
        push_edges(edges, kuzu_conn, repo_name)


def purge_repository(repo_name: str, db_path: str = None):
    """Delete ALL records for this repository from PostgreSQL and Kùzu."""
    print(f"🧹 Purging repository '{repo_name}' ...")

    # PostgreSQL — delegate to initialize_db.clear_repo() rather than
    # re-implementing the delete here: it uses the actual `repository_name`
    # column and also clears key_points, which has no FK to `files` and so
    # was never touched by a bare "DELETE FROM files" cascade.
    conn = get_connection()
    try:
        counts = clear_repo(conn, repo_name)
        print(f"  ✅ Removed {counts['files']} files, {counts['business_rules']} business rules, "
              f"{counts['classes']} classes, {counts['functions']} functions, "
              f"{counts['key_points']} key points from PostgreSQL.")
    except Exception as e:
        print(f"  ❌ PostgreSQL purge failed: {e}")
    finally:
        conn.close()

    # Kùzu
    try:
        ingester = GraphIngester(repo_name, db_path)
        ingester.clear_graph()
        ingester.close()
        print(f"  ✅ Cleared Kùzu graph for '{repo_name}'.")
    except Exception as e:
        print(f"  ❌ Kùzu purge failed: {e}")

    print("✅ Purge complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — full repository ingestion pipeline"
    )
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("--from", dest="from_stage", default="init",
                        choices=["init", "parse", "extract", "synthesize", "graph", "deps", "embeddings"],
                        help="Start from this stage")
    parser.add_argument("--only", dest="only_stage", default=None,
                        choices=["init", "parse", "extract", "synthesize", "graph", "deps", "embeddings"],
                        help="Run only this stage")
    parser.add_argument("--db", default=None, help="Kùzu database path")
    parser.add_argument("--purge", action="store_true",
                        help="Purge all data for this repository before running")

    # LLM provider/model
    parser.add_argument("--provider", default="openai",
                        help="LLM provider (openai, anthropic, deepseek, etc.)")
    parser.add_argument("--model", default="gpt-4o",
                        help="LLM model name (e.g., gpt-4o, claude-3-opus)")

    # Extraction options
    parser.add_argument("--extract-workers", type=int, default=1,
                        help="Workers for business‑rule extraction")
    parser.add_argument("--extract-rate-limit", type=int, default=0,
                        help="Max LLM calls per minute (0 = unlimited)")
    parser.add_argument("--extract-lines", type=int, default=80,
                        help="Max lines to read per file for extraction")
    parser.add_argument("--only-unprocessed", action="store_true",
                        help="Skip files already processed (resume extraction)")

    # Embeddings options
    parser.add_argument("--embeddings-workers", type=int, default=4,
                        help="Workers for BERTScore (if using method=bertscore)")
    parser.add_argument("--embeddings-method", choices=["fast", "bertscore"], default="fast",
                        help="Embedding method: fast (matrix multiply) or bertscore (per-pair)")
    parser.add_argument("--embeddings-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model for --method fast")
    parser.add_argument("--embeddings-batch-size", type=int, default=64,
                        help="Batch size for --method fast")
    parser.add_argument("--embeddings-no-cache", action="store_true",
                        help="Disable caching for BERTScore method")
    parser.add_argument("--embeddings-sequential", action="store_true",
                        help="Use sequential processing for BERTScore method")
    parser.add_argument("--embeddings-collection", default=None,
                        help="Override ChromaDB collection name")

    # Toggle stages
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip business‑rule extraction")
    parser.add_argument("--no-synthesize", action="store_true",
                        help="Skip key‑point synthesis")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip ChromaDB embedding stage")

    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    if not repo_path.exists():
        print(f"Error: {repo_path} not found.")
        sys.exit(1)

    repo_name = repo_path.name

    # Purge if requested
    if args.purge:
        purge_repository(repo_name, args.db)
        if args.only_stage is None and args.from_stage == "init":
            return

    # Init DB
    if args.only_stage == "init" or (args.only_stage is None and args.from_stage == "init"):
        init_db()
        if args.only_stage:
            return

    # Get file list (needed for graph/deps, but parse stage populates DB)
    files = get_file_list(repo_path)
    print(f"  Found {len(files)} files.")

    # Stages order
    all_stages = ["parse", "extract", "synthesize", "embeddings", "graph", "deps"]
    start_idx = 0 if args.from_stage == "init" else all_stages.index(args.from_stage) if args.from_stage in all_stages else 0
    end_idx = len(all_stages) - 1 if args.only_stage is None else all_stages.index(args.only_stage)

    for stage in all_stages[start_idx:end_idx+1]:
        print(f"\n--- Running stage: {stage} ---")

        if stage == "parse":
            store_detailed_metrics(repo_path, files)

        elif stage == "extract":
            if args.no_extract:
                print("  Skipping extraction (--no-extract).")
            else:
                run_extraction(
                    repo_name=repo_name,
                    repo_path=str(repo_path),
                    max_lines=args.extract_lines,
                    only_unprocessed=args.only_unprocessed,
                    workers=args.extract_workers,
                    rate_limit=args.extract_rate_limit,
                    provider=args.provider,
                    model=args.model,
                    synthesize=False   # we handle synthesis separately
                )

        elif stage == "synthesize":
            if args.no_synthesize:
                print("  Skipping synthesis (--no-synthesize).")
            else:
                run_synthesis(
                    repo_name=repo_name,
                    provider=args.provider,
                    model=args.model,
                    max_key_points=15
                )

        elif stage == "embeddings":
            if args.no_embeddings:
                print("  Skipping embeddings (--no-embeddings).")
            else:
                # Determine chroma db path (default: ./chroma_db)
                chroma_path = "./chroma_db"
                run_embeddings(
                    repo_name=repo_name,
                    chroma_db_path=chroma_path,
                    collection_name=args.embeddings_collection,
                    workers=args.embeddings_workers,
                    no_cache=args.embeddings_no_cache,
                    sequential=args.embeddings_sequential,
                    method=args.embeddings_method,
                    embed_model=args.embeddings_model,
                    batch_size=args.embeddings_batch_size,
                )

        elif stage == "graph":
            build_graph(repo_name, files, args.db)

        elif stage == "deps":
            build_dependencies(repo_name, repo_path, files, args.db)

    print("\n✅ Ingestion complete.")


if __name__ == "__main__":
    main()