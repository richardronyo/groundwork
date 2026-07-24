#!/usr/bin/env python3
"""
Groundwork — Multi-Database Context Retrieval (LangChain)

Wraps the three knowledge-base stores as a single LangChain retriever:
  1. BERTScore matches the query against key points (from PostgreSQL)
  2. ChromaDB supplies the top files for the best key point
  3. Kùzu supplies DEPENDS_ON dependencies + directory location
  4. PostgreSQL supplies business rules

The result is exposed as a LangChain BaseRetriever returning Documents,
so it drops straight into RetrievalQA chains, agents, or any LCEL pipeline.

Usage:
    python3 grab_context.py "How does user authentication work?"
    python3 grab_context.py "payment processing" --top 3
    python3 grab_context.py "session handling" --ask   # full RAG answer

Requirements:
    pip install langchain langchain-core langchain-openai \
                chromadb psycopg kuzu bert-score python-dotenv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chromadb
import psycopg
from kb.graph.kuzu_store import (
    get_connection as get_kuzu_connection,
    get_dependencies as kuzu_get_dependencies,
)
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH    = "./chroma_db"
CHROMA_COLLECTION = "groundwork"


def collection_name_for(repo_name: str) -> str:
    """Per-repo ChromaDB collection name (matches embeddings.py)."""
    safe = "".join(c if c.isalnum() else "_" for c in (repo_name or "").lower())
    return f"{CHROMA_COLLECTION}_{safe}"
BERT_MODEL        = "distilbert-base-uncased"

DB_CONFIG = {
    "host": "localhost",
    "dbname": "repo_analysis",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD"),
}




# ── Helpers that read each store (reused by the retriever) ────────────────────

def load_bert_scorer():
    try:
        from bert_score import score as bert_score_fn
        return bert_score_fn
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        sys.exit(1)


def list_repos() -> List[str]:
    """All repositories present in the knowledge base."""
    conn = psycopg.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT repository_name FROM files ORDER BY repository_name")
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def resolve_repo_name(repo_name: str = None) -> Optional[str]:
    """If repo_name is given, use it. If not and only one repo exists, use that.
    If multiple exist and none specified, return None (caller should prompt)."""
    if repo_name:
        return repo_name
    repos = list_repos()
    if len(repos) == 1:
        return repos[0]
    return None


def load_key_points_from_db(repo_name: str = None) -> List[str]:
    """Key points live in PostgreSQL, scoped by repository."""
    conn = psycopg.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            if repo_name is None:
                cur.execute("SELECT DISTINCT repository_name FROM key_points LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return []
                repo_name = row[0]
            cur.execute(
                "SELECT point_text FROM key_points WHERE repository_name = %s ORDER BY point_index",
                (repo_name,),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def score_prompt_against_keypoints(prompt, key_points, bert_score_fn) -> List[float]:
    cands = [prompt] * len(key_points)
    _, _, F1 = bert_score_fn(cands=cands, refs=key_points,
                             model_type=BERT_MODEL, verbose=False)
    return [float(s.item()) for s in F1]


def get_files_for_keypoint(key_point: str, top_n: int, repo_name: str = None) -> List[Dict]:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll_name = collection_name_for(repo_name)
    try:
        collection = client.get_collection(coll_name)
    except Exception:
        # Fall back to the legacy single collection if the per-repo one is absent
        collection = client.get_collection(CHROMA_COLLECTION)
    results = collection.get(include=["documents", "metadatas"])

    matching = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if meta.get("top_kp") == key_point:
            matching.append({"doc": doc, "meta": meta})
    matching.sort(key=lambda x: x["meta"].get("top_score", 0), reverse=True)
    return matching[:top_n]


def get_dependencies(file_paths: List[str], repo_name: str = None) -> Dict[str, List[str]]:
    """Dependency lookup from the embedded Kùzu graph (repo-scoped)."""
    if not file_paths:
        return {}
    try:
        conn = get_kuzu_connection()
        return kuzu_get_dependencies(conn, file_paths, repo_name)
    except Exception as e:
        print(f"  (graph unavailable: {e})")
        return {}


def get_business_rules(file_paths: List[str], repo_name: str = None) -> Dict[str, List[str]]:
    if not file_paths:
        return {}
    conn = psycopg.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT f.file_path, array_agg(br.rule_text ORDER BY br.id) AS rules
                FROM files f
                JOIN business_rules br ON br.file_id = f.id
                WHERE f.file_path = ANY(%s)
                AND (%s::text IS NULL OR f.repository_name = %s) 
                GROUP BY f.file_path
            """, (file_paths, repo_name, repo_name))
            return {row[0]: (row[1] or []) for row in cur.fetchall()}
    finally:
        conn.close()


def get_repo_metrics(repo_name: str = None) -> Dict:
    """
    Pulls aggregate structural stats from PostgreSQL to ground the inference:
    total files, total lines, language breakdown, top-level directories, and
    overall class/function/method counts.
    """
    conn = psycopg.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            if repo_name is None:
                cur.execute("SELECT DISTINCT repository_name FROM files LIMIT 1")
                row = cur.fetchone()
                if not row:
                    return {}
                repo_name = row[0]

            # Totals
            cur.execute("""
                SELECT COUNT(*), COALESCE(SUM(lines),0),
                       COALESCE(SUM(classes),0), COALESCE(SUM(functions),0),
                       COALESCE(SUM(methods),0), COALESCE(SUM(async_functions),0)
                FROM files WHERE repository_name = %s
            """, (repo_name,))
            total_files, total_lines, classes, functions, methods, async_fns = cur.fetchone()

            # Language breakdown
            cur.execute("""
                SELECT language, COUNT(*) AS n, COALESCE(SUM(lines),0) AS loc
                FROM files WHERE repository_name = %s
                GROUP BY language ORDER BY n DESC
            """, (repo_name,))
            languages = [
                {"language": r[0], "files": r[1], "lines": r[2]}
                for r in cur.fetchall()
            ]

            # Top-level directories (first path segment) by file count
            cur.execute("""
                SELECT split_part(file_path, '/', 1) AS top_dir, COUNT(*) AS n
                FROM files WHERE repository_name = %s
                GROUP BY top_dir ORDER BY n DESC LIMIT 12
            """, (repo_name,))
            top_dirs = [{"dir": r[0], "files": r[1]} for r in cur.fetchall()]

        return {
            "repository_name": repo_name,
            "total_files": total_files,
            "total_lines": total_lines,
            "classes": classes,
            "functions": functions,
            "methods": methods,
            "async_functions": async_fns,
            "languages": languages,
            "top_directories": top_dirs,
        }
    finally:
        conn.close()


def format_metrics_block(metrics: Dict) -> str:
    """Renders repo metrics as a compact text block for the LLM prompt."""
    if not metrics:
        return ""

    lines = [f"Repository structural profile for '{metrics['repository_name']}':"]
    lines.append(f"- Total files: {metrics['total_files']}")
    lines.append(f"- Total lines of code: {metrics['total_lines']}")
    lines.append(f"- Classes: {metrics['classes']}, Functions: {metrics['functions']}, "
                 f"Methods: {metrics['methods']}, Async functions: {metrics['async_functions']}")

    if metrics["languages"]:
        lang_str = ", ".join(
            f"{l['language']} ({l['files']} files, {l['lines']} loc)"
            for l in metrics["languages"][:8]
        )
        lines.append(f"- Languages: {lang_str}")

    if metrics["top_directories"]:
        dir_str = ", ".join(
            f"{d['dir']} ({d['files']})" for d in metrics["top_directories"][:10]
        )
        lines.append(f"- Top-level directories: {dir_str}")

    return "\n".join(lines)


# ── Repo-level question detection ─────────────────────────────────────────────

REPO_LEVEL_PATTERNS = [
    "what is this", "what does this", "what's this",
    "what is the codebase", "what does the codebase",
    "what is the repo", "what does the repo", "what is the repository",
    "what is the project", "what does the project",
    "overall", "in general", "high level", "high-level",
    "purpose of", "simulating", "simulate", "about this",
    "summarize", "summary of", "describe the", "what kind of",
    "what type of", "overview",
]


def is_repo_level_question(query: str) -> bool:
    """
    Heuristic: does the query ask about the whole codebase rather than a
    specific feature? Repo-level questions are answered by the key points
    collectively, not by retrieving individual files.
    """
    q = query.lower().strip()
    return any(pat in q for pat in REPO_LEVEL_PATTERNS)


def get_all_key_points_as_document(query: str, repo_name: str = None) -> Document:
    """
    Builds a single Document combining two kinds of evidence for repo-level
    inference:
      1. The structural profile (file counts, languages, directories) from the
         files table — concrete grounding.
      2. The synthesized key points — behavioral grounding.
    Together these let the LLM infer what the codebase actually is.
    """
    metrics = get_repo_metrics(repo_name)
    key_points = load_key_points_from_db(repo_name)

    metrics_block = format_metrics_block(metrics)
    kp_block = (
        "Repository capability catalog (what this codebase does as a whole):\n\n" +
        "\n".join(f"- {kp}" for kp in key_points)
    )

    page_content = (metrics_block + "\n\n" + kp_block) if metrics_block else kp_block

    return Document(
        page_content=page_content,
        metadata={
            "retrieval_type": "repo_level",
            "key_point_count": len(key_points),
            "matched_key_point": "(all key points + structural profile)",
            "key_point_score": 1.0,
            "name": "repository_overview",
            "relative": "(whole repository)",
            "score": 1.0,
            "dependencies": [],
            "rule_count": 0,
            "total_files": metrics.get("total_files", 0) if metrics else 0,
        },
    )


# ── The LangChain Retriever ───────────────────────────────────────────────────

class GroundworkRetriever(BaseRetriever):
    """
    A LangChain retriever over the Groundwork knowledge base.

    For a query it returns one Document per relevant file, where:
      - page_content = the file's business rules (the LLM-facing context)
      - metadata     = file path, score, dependencies, matched key point

    Because it subclasses BaseRetriever, it works with RetrievalQA,
    create_retrieval_chain, agents, and any LCEL `retriever | prompt | llm`.
    """

    top_n: int = Field(default=2)
    min_score: float = Field(default=0.3)
    repo_name: Optional[str] = Field(default=None)
    mode: str = Field(default="auto")  # "auto" | "repo" | "file"
    _bert_fn: object = None

    def _ensure_bert(self):
        if self._bert_fn is None:
            object.__setattr__(self, "_bert_fn", load_bert_scorer())
        return self._bert_fn

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:

        # Resolve which repo we are querying (may be None if ambiguous)
        repo = resolve_repo_name(self.repo_name)

        # ── Route: repo-level vs file-level ────────────────────────────────
        if self.mode == "repo" or (self.mode == "auto" and is_repo_level_question(query)):
            return [get_all_key_points_as_document(query, repo)]

        # ── File-level retrieval (original flow) ───────────────────────────
        # 1. Key points from Postgres (repo-scoped)
        key_points = load_key_points_from_db(repo)
        if not key_points:
            return []

        # 2. BERTScore the query against key points
        bert_fn = self._ensure_bert()
        scores = score_prompt_against_keypoints(query, key_points, bert_fn)
        ranked = sorted(zip(key_points, scores), key=lambda x: x[1], reverse=True)

        best_kp, best_score = ranked[0]
        if best_score < self.min_score:
            return []

        # 3. ChromaDB → top files for the best key point
        file_matches = get_files_for_keypoint(best_kp, self.top_n, repo)
        if not file_matches:
            return []

        file_paths = [m["meta"]["relative"] for m in file_matches]

        # 4. Kùzu dependencies + 5. Postgres rules
        dependencies = get_dependencies(file_paths, repo)
        rules        = get_business_rules(file_paths, repo)

        # 6. Build one Document per file
        docs = []
        for m in file_matches:
            meta = m["meta"]
            rel  = meta["relative"]
            file_rules = rules.get(rel, []) or (m["doc"].split("\n\n") if m["doc"] else [])

            page_content = (
                f"File: {rel}\n\n"
                f"Business rules:\n" +
                "\n".join(f"- {r}" for r in file_rules)
            )

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "relative": rel,
                    "name": meta.get("name"),
                    "score": meta.get("top_score", 0),
                    "matched_key_point": best_kp,
                    "key_point_score": best_score,
                    "dependencies": dependencies.get(rel, []),
                    "rule_count": len(file_rules),
                },
            ))
        return docs


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_documents(docs: List[Document]):
    if not docs:
        print("\n  No relevant context found (no key point passed the score threshold).")
        print("  Tip: for whole-codebase questions try phrasing like "
              "'what does this codebase do', or pass --mode repo.")
        return

    # Repo-level result — a single overview document
    if docs[0].metadata.get("retrieval_type") == "repo_level":
        m = docs[0].metadata
        print(f"\n  Retrieval type: REPO-LEVEL (whole codebase)")
        print(f"  Answered from {m['key_point_count']} key points.\n")
        print(docs[0].page_content)
        print()
        return

    # File-level result
    print(f"\n  Retrieval type: FILE-LEVEL")
    print(f"  Retrieved {len(docs)} documents:\n")
    print(f"  Matched key point: {docs[0].metadata['matched_key_point'][:80]}...")
    print(f"  Key point score  : {docs[0].metadata['key_point_score']:.4f}\n")
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        print(f"  ── Document {i}: {m['name']} ──")
        print(f"     Path        : {m['relative']}")
        print(f"     File score  : {m['score']:.4f}")
        print(f"     Rules       : {m['rule_count']}")
        print(f"     Dependencies: {len(m['dependencies'])}")
        for dep in m["dependencies"][:3]:
            print(f"        → {Path(dep).name}")
        print()


# ── Prompts for the answer step ───────────────────────────────────────────────

REPO_LEVEL_SYSTEM = """You are a senior software architect.
You are given two kinds of evidence about an entire codebase:
  1. A STRUCTURAL PROFILE — file counts, languages, lines of code, and the
     top-level directory layout.
  2. A CAPABILITY CATALOG — business rules and behaviors synthesized from the code.

Your job is to INFER and explain what this software actually is — the kind of
application or system it implements, its core domain, its likely architecture,
and what problem it solves.

Reason across BOTH the structure and the capabilities. Use the structural profile
for concrete grounding (e.g. "predominantly C# across Libraries/Plugins/Presentation
suggests a large layered .NET application") and the capabilities for behavioral
grounding (e.g. "rules about carts, orders, and stock indicate e-commerce").

Do not just list things back. Form a coherent, specific conclusion about what is
being built, as if explaining it to a new engineer, and justify it from the
evidence. If the evidence points to a particular type of system, name it."""

FILE_LEVEL_SYSTEM = """You are a software analyst. Answer the question using ONLY
the provided business-rule context. Cite the file names you draw from. If the
context does not contain the answer, say so plainly."""


def run_rag(query: str, docs: List[Document]):
    """Feed retrieved context to an LLM. Uses a synthesis prompt for repo-level
    questions (infer what the system IS) and a grounded prompt for file-level."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        print("Install langchain-openai for --ask: pip install langchain-openai")
        return

    is_repo_level = docs and docs[0].metadata.get("retrieval_type") == "repo_level"

    context = "\n\n---\n\n".join(d.page_content for d in docs)

    if is_repo_level:
        system = REPO_LEVEL_SYSTEM
        human = ("Here is the capability catalog for the codebase:\n\n{context}\n\n"
                 "Question: {question}\n\n"
                 "Based on these capabilities, infer and explain what this codebase is.")
    else:
        system = FILE_LEVEL_SYSTEM
        human = "Context:\n{context}\n\nQuestion: {question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])
    llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))
    chain = prompt | llm
    resp = chain.invoke({"context": context, "question": query})

    print("\n  ── LLM Inference ──\n" if is_repo_level else "\n  ── LLM Answer ──\n")
    print(resp.content)


def main():
    parser = argparse.ArgumentParser(
        description="Groundwork context retrieval via LangChain"
    )
    parser.add_argument("prompt", nargs="?", help="Query prompt")
    parser.add_argument("--top", type=int, default=2, help="Top files to retrieve")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--repo", default=None,
                        help="Repository name to query (required if multiple repos exist)")
    parser.add_argument("--list-repos", action="store_true",
                        help="List repositories in the knowledge base and exit")
    parser.add_argument("--mode", default="auto", choices=["auto", "repo", "file"],
                        help="auto-detect (default), force repo-level, or force file-level")
    parser.add_argument("--ask", action="store_true",
                        help="Also generate an LLM answer from the retrieved context")
    args = parser.parse_args()

    if args.list_repos:
        repos = list_repos()
        if repos:
            print("\n  Repositories in the knowledge base:")
            for r in repos:
                print(f"    - {r}")
            print()
        else:
            print("\n  No repositories found. Run the ingestion pipeline first.\n")
        return

    if not args.prompt:
        parser.print_help()
        return

    # Resolve repo; if ambiguous and not specified, tell the user to pick one
    resolved = resolve_repo_name(args.repo)
    if resolved is None:
        repos = list_repos()
        print("\n  Multiple repositories exist — specify one with --repo:")
        for r in repos:
            print(f"    - {r}")
        print("\n  Example: python grab_context.py \"your question\" --repo flask\n")
        return
    args.repo = resolved
    print(f"  Querying repository: {args.repo}")

    retriever = GroundworkRetriever(
        top_n=args.top,
        min_score=args.min_score,
        repo_name=args.repo,
        mode=args.mode,
    )

    # This is the LangChain entry point — same call a chain would make
    docs = retriever.invoke(args.prompt)

    # Repo-level questions are inference questions — they always want the LLM
    # to reason about what the codebase is, so run the inference automatically.
    is_repo_level = docs and docs[0].metadata.get("retrieval_type") == "repo_level"

    if is_repo_level:
        run_rag(args.prompt, docs)
    else:
        print_documents(docs)
        if args.ask and docs:
            run_rag(args.prompt, docs)


if __name__ == "__main__":
    main()