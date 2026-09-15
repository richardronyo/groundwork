"""
Groundwork — RAG context assembly for test generation.

Retrieves three kinds of context for a target file, reusing existing
pipeline outputs rather than re-deriving anything:
  - top business rules already extracted for this file (business_rules table)
  - the file's top-matching key point, already computed by the embeddings
    stage and stored as `top_kp` metadata on its Chroma entry — this is a
    direct metadata lookup, not a fresh similarity search, because that
    collection's vectors are per-file similarity-to-keypoint scores (see
    embeddings.py), not a general embedding space you can query with an
    arbitrary text vector
  - source of files this target file actually depends on, via Kùzu
    DEPENDS_ON edges from the `deps` stage, so the model sees real imported
    code instead of guessing what an import provides

Key/path convention: business_rules, the Chroma "groundwork_<repo>"
collection, and Kùzu are all keyed by the repo-prefixed relative path
("<repo_name>/<path>", per tree_to_json.py) — the same convention that has
caused the doubled-prefix bug elsewhere in this project. target_file here is
filesystem-relative-to-repo-root (no prefix), matching generate_test_simple.py's
convention, so every lookup below explicitly builds the prefixed DB key
rather than assuming the two forms are interchangeable.
"""

from pathlib import Path

from kb.relationaldb.initialize_db import get_connection, load_business_rules_from_db
from kb.vector.embeddings import get_collection, collection_name_for, CHROMA_DB_PATH
from kb.graph.kuzu_store import get_reader, get_dependencies, KUZU_DB_PATH

MAX_DEP_CHARS = 2000  # keep each dependency snippet bounded, same spirit as MAX_READ_CHARS


def _db_key(repo_name: str, target_file: str) -> str:
    """Builds the repo-prefixed key that business_rules/Chroma/Kùzu actually
    use, regardless of whether target_file was already given with or without
    the prefix (the same doubled-prefix tolerance as generate_test_simple.py,
    just going the other direction: adding the prefix instead of stripping it)."""
    prefix = f"{repo_name}/"
    return target_file if target_file.startswith(prefix) else f"{prefix}{target_file}"


def _top_business_rules(repo_name: str, db_key: str, top_n: int) -> list[str]:
    conn = get_connection()
    try:
        rules_by_file = load_business_rules_from_db(conn, repo_name)
    finally:
        conn.close()
    return rules_by_file.get(db_key, [])[:top_n]


def _top_key_point(repo_name: str, db_key: str, chroma_db_path: str) -> str | None:
    collection = get_collection(chroma_db_path, collection_name_for(repo_name))
    if collection.count() == 0:
        return None
    got = collection.get(ids=[db_key], include=["metadatas"])
    metas = got.get("metadatas") or []
    return metas[0].get("top_kp") if metas else None


def _dependency_sources(repo_root: Path, repo_name: str, db_key: str,
                         top_n: int, kuzu_db_path: str | None) -> list[dict]:
    try:
        conn = get_reader(kuzu_db_path or KUZU_DB_PATH)
    except RuntimeError:
        return []  # no graph built yet (deps stage not run) — degrade gracefully

    deps = get_dependencies(conn, [db_key], repo_name).get(db_key, [])[:top_n]

    prefix = f"{repo_name}/"
    sources = []
    for dep_key in deps:
        rel_within_repo = dep_key[len(prefix):] if dep_key.startswith(prefix) else dep_key
        full_path = (repo_root / rel_within_repo).resolve()
        if full_path.exists():
            sources.append({
                "path": dep_key,
                "content": full_path.read_text(errors="replace")[:MAX_DEP_CHARS],
            })
    return sources


def gather_rag_context(repo_root: Path, repo_name: str, target_file: str,
                        top_n_rules: int = 5, top_n_deps: int = 3,
                        chroma_db_path: str = CHROMA_DB_PATH,
                        kuzu_db_path: str | None = None) -> dict:
    """
    Returns:
        {
            "rules": [str, ...],
            "top_key_point": str | None,
            "dependencies": [{"path": str, "content": str}, ...],
        }
    Every piece degrades to empty/None rather than raising if that stage of
    the pipeline (extract/embeddings/deps) hasn't been run yet for this repo —
    RAG context is an enhancement, not a hard requirement for generation.
    """
    db_key = _db_key(repo_name, target_file)
    return {
        "rules": _top_business_rules(repo_name, db_key, top_n_rules),
        "top_key_point": _top_key_point(repo_name, db_key, chroma_db_path),
        "dependencies": _dependency_sources(repo_root, repo_name, db_key, top_n_deps, kuzu_db_path),
    }


def format_rag_context(context: dict) -> str:
    """Renders gather_rag_context()'s output as a prompt-ready text block.
    Returns "" if nothing was retrieved, so callers can skip the section
    entirely rather than injecting an empty header."""
    parts = []

    if context["rules"]:
        parts.append("Known business rules for this file:\n" +
                      "\n".join(f"- {r}" for r in context["rules"]))

    if context["top_key_point"]:
        parts.append(f"Most relevant repository-wide key point for this file: "
                      f"{context['top_key_point']}")

    if context["dependencies"]:
        dep_blocks = [f"--- {dep['path']} ---\n{dep['content']}" for dep in context["dependencies"]]
        parts.append("Relevant imported files:\n\n" + "\n\n".join(dep_blocks))

    return "\n\n".join(parts)