"""
Groundwork — the ingestion pipeline as one LangGraph graph.

    scan ─► parse ─► deps ─► extract ─► synthesize ─► embed ─► translate
     │        │        │        │            │           │         │
     └────────┴────────┴── any stage that fails stops the run ─────┴─► END

  scan        walk the repo → File/Directory nodes + CONTAINS/SAME_DIR in Kùzu
  parse       tree-sitter → files / classes / functions rows in Postgres
  deps        import statements → DEPENDS_ON edges in Kùzu
  extract     the extraction graph (kb/vector/extract_business_rules.py) as a
              subgraph: per-file Send fan-out, optional interrogation loop
  synthesize  map-reduce key points → key_points
  embed       per-file key-point similarity vectors → ChromaDB
  translate   the translation graph (kb/translate/graph.py) for each file in
              --translate, in parallel, reading everything the stages above built

Only the stages you select are added to the graph (--only / --skip), in this
order. Each stage node wraps the module that already does that job, so every
module still runs on its own from its own CLI.

Usage:
    python3 -m kb.pipeline.ingest_graph repos/flask
    python3 -m kb.pipeline.ingest_graph repos/flask --only scan,parse,deps
    python3 -m kb.pipeline.ingest_graph repos/flask --only extract,synthesize --refine-rounds 2 --workers 8
    python3 -m kb.pipeline.ingest_graph repos/shop --skip embed --translate orders.py --translate-to javascript
    python3 -m kb.pipeline.ingest_graph repos/flask --checkpoint cache/runs.sqlite --thread-id flask-oct19
"""

from __future__ import annotations

import argparse
import gc
import operator
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from kb.llm.client import run_config, llm_from_config, DEFAULT_MODEL
from kb.graph.tree_to_json import files_from_walk

STAGES = ["scan", "parse", "deps", "extract", "synthesize", "embed", "translate"]
DEFAULT_STAGES = ["scan", "parse", "deps", "extract", "synthesize", "embed"]

CODE_LANGUAGES = {
    "Python", "JavaScript", "TypeScript", "JavaScript (React)", "TypeScript (React)",
    "Java", "Kotlin", "C#", "C++", "C", "C/C++ Header", "Go", "Rust", "Ruby", "PHP",
    "Swift", "Shell", "Batch", "SQL", "R",
}


class PipelineState(TypedDict, total=False):
    # inputs
    repo_path: str
    repo_name: str
    clear: bool
    dep_lines: int
    max_lines: int
    only_unprocessed: bool
    refine_rounds: int
    max_key_points: int
    embed_method: str
    translate_files: list[str]
    translate_to: str | None
    translate_rounds: int
    # outputs
    files: list[dict]
    stage_log: Annotated[list[dict], operator.add]     # {stage, ok, seconds, detail}
    errors: Annotated[list[str], operator.add]
    translations: Annotated[list[dict], operator.add]


# ── Kùzu: one writer per process ──────────────────────────────────────────────
# Kùzu allows a single read-write Database at a time, and scan + deps both
# write. Share one connection between them, then release it so later stages
# (translate, RAG) can open the graph read-only.

_kuzu = {"conn": None}


def _kuzu_writer():
    if _kuzu["conn"] is None:
        from kb.graph.kuzu_store import get_connection
        _kuzu["conn"] = get_connection()
    return _kuzu["conn"]


def _release_kuzu():
    if _kuzu["conn"] is not None:
        conn = _kuzu["conn"]
        _kuzu["conn"] = None
        db = getattr(conn, "_db", None)
        for obj in (conn, db):
            close = getattr(obj, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass
        del conn, db
        gc.collect()


def _code_files(state: PipelineState) -> list[dict]:
    files = state.get("files") or files_from_walk(state["repo_path"], state["repo_name"])
    return [f for f in files if f.get("language") in CODE_LANGUAGES]


def _stage(name):
    """Decorator: timing, logging, and turning exceptions into a recorded
    error that stops the run (instead of a traceback mid-graph)."""
    def wrap(fn):
        def node(state, config):
            print(f"\n══ {name} " + "═" * (60 - len(name)))
            t0 = time.time()
            try:
                update = fn(state, config) or {}
                detail = update.pop("_detail", "")
                log = {"stage": name, "ok": True, "seconds": round(time.time() - t0, 1),
                       "detail": detail}
                return {**update, "stage_log": [log]}
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")
                return {"errors": [f"{name}: {e}"],
                        "stage_log": [{"stage": name, "ok": False,
                                       "seconds": round(time.time() - t0, 1), "detail": str(e)}]}
        node.__name__ = name
        return node
    return wrap


# ── stages ────────────────────────────────────────────────────────────────────

@_stage("scan")
def scan(state: PipelineState, config) -> dict:
    from kb.graph.json_to_graph import GraphIngester
    all_files = files_from_walk(state["repo_path"], state["repo_name"])
    files = [f for f in all_files if f["language"] in CODE_LANGUAGES]
    print(f"  {len(all_files)} files on disk, {len(files)} code files")
    ingester = GraphIngester(state["repo_name"], conn=_kuzu_writer())
    if state.get("clear"):
        ingester.clear_graph()
    ingester.ingest(files)
    return {"files": all_files, "_detail": f"{len(files)} code files"}


@_stage("parse")
def parse(state: PipelineState, config) -> dict:
    from kb.relationaldb.initialize_db import (
        init_db, get_connection, save_file, save_structure, clear_repo)
    from kb.parser.parser import (
        get_parser, extract_file_metrics, reshape_for_db, structure_for_db)
    from kb.parser.parser_dicts import EXTENSION_TO_MODULE, MISSING_GRAMMARS

    init_db()
    files = _code_files(state)
    disk_base = Path(state["repo_path"]).resolve().parent     # relative starts with repo name
    parsers, parsed, plain, failed, missing = {}, 0, 0, [], set()
    zero = {"classes": 0, "functions": 0, "methods": 0, "async_functions": 0, "imports": 0}

    conn = get_connection()
    try:
        if state.get("clear"):
            clear_repo(conn, state["repo_name"])
        for i, f in enumerate(files, 1):
            path = disk_base / f["relative"]
            ext = "." + f["extension"] if f["extension"] else ""
            try:
                source = path.read_bytes()
                if ext in EXTENSION_TO_MODULE:
                    parser = parsers.get(ext) or parsers.setdefault(ext, get_parser(str(path)))
                    metrics = extract_file_metrics(parser.parse(source).root_node, ext, source)
                    file_id = save_file(conn, state["repo_name"], f["relative"], f["language"],
                                        reshape_for_db(metrics), metrics["imports"])
                    classes, functions = structure_for_db(metrics)
                    save_structure(conn, file_id, classes, functions)
                    parsed += 1
                else:
                    # No grammar: still record the file so extraction sees it.
                    if ext in MISSING_GRAMMARS:
                        missing.add(MISSING_GRAMMARS[ext])
                    lines = len(source.decode(errors="replace").splitlines())
                    save_file(conn, state["repo_name"], f["relative"], f["language"],
                              {**zero, "lines": lines}, [])
                    plain += 1
                conn.commit()
            except Exception as e:
                conn.rollback()
                failed.append(f"{f['relative']}: {e}")
            pct = int(i / len(files) * 40) if files else 40
            print(f"\r  [{'█' * pct}{'░' * (40 - pct)}] {i}/{len(files)}", end="", flush=True)
    finally:
        conn.close()
    print(f"\n  {parsed} parsed with tree-sitter, {plain} recorded without a grammar, "
          f"{len(failed)} failed")
    if missing:
        print(f"  Grammars not installed: {', '.join(sorted(missing))}")
    for msg in failed[:10]:
        print(f"    ! {msg}")
    return {"_detail": f"{parsed} parsed, {plain} without grammar, {len(failed)} failed"}


@_stage("deps")
def deps(state: PipelineState, config) -> dict:
    from kb.graph.file_dependencies import extract_dependencies, push_edges, print_summary
    files = _code_files(state)
    edges = extract_dependencies(files, Path(state["repo_path"]),
                                 n_lines=state.get("dep_lines", 20))
    print(f"  Resolved {len(edges)} dependencies between known files.")
    print_summary(edges)
    if edges:
        push_edges(edges, _kuzu_writer(), state["repo_name"])
    return {"_detail": f"{len(edges)} DEPENDS_ON edges"}


@_stage("extract")
def extract(state: PipelineState, config) -> dict:
    from kb.vector.extract_business_rules import build_extraction_graph
    _release_kuzu()
    out = build_extraction_graph().invoke({
        "repo_name": state["repo_name"], "repo_path": state["repo_path"],
        "max_lines": state.get("max_lines", 80),
        "only_unprocessed": state.get("only_unprocessed", False),
        "refine_rounds": state.get("refine_rounds", 0),
        "synthesize": False, "results": [],
    }, config)
    results = out.get("results", [])
    ok = sum(r["ok"] for r in results)
    if results and not ok:
        raise RuntimeError("every file failed extraction — see the errors above")
    return {"_detail": f"{ok}/{len(results)} files, {sum(r['n_rules'] for r in results)} rules"}


@_stage("synthesize")
def synthesize(state: PipelineState, config) -> dict:
    from kb.relationaldb.initialize_db import (
        get_connection, load_business_rules_from_db, save_key_points)
    from kb.vector.extract_business_rules import synthesize_repo_function, MAX_KEY_POINTS
    conn = get_connection()
    try:
        rules = load_business_rules_from_db(conn, state["repo_name"])
        if not rules:
            raise RuntimeError("no business rules in the DB — run the extract stage first")
        points = synthesize_repo_function(
            rules, max_points=state.get("max_key_points", MAX_KEY_POINTS),
            llm=llm_from_config(config, temperature=0.3))
        save_key_points(conn, state["repo_name"], points)
        conn.commit()
    finally:
        conn.close()
    return {"_detail": f"{len(points)} key points"}


@_stage("embed")
def embed(state: PipelineState, config) -> dict:
    try:
        from kb.vector.embeddings import run_embeddings
    except ImportError as e:
        raise RuntimeError(f"embeddings need chromadb + sentence-transformers ({e})")
    run_embeddings(state["repo_name"], method=state.get("embed_method", "fast"))
    return {"_detail": "vectors written"}


def _translate_fan_out(state: PipelineState):
    """Runs after the `translate` gate node: one Send per requested file."""
    if state.get("errors") or not state.get("translate_files") or not state.get("translate_to"):
        return END
    return [Send("translate_file", {"repo_path": state["repo_path"],
                                    "repo_name": state["repo_name"], "file": f,
                                    "to": state["translate_to"],
                                    "rounds": state.get("translate_rounds", 5)})
            for f in state["translate_files"]]


@_stage("translate")
def translate_gate(state: PipelineState, config) -> dict:
    _release_kuzu()
    files = state.get("translate_files") or []
    if not files or not state.get("translate_to"):
        print("  Nothing to translate (pass --translate FILE ... --translate-to LANG).")
    else:
        print(f"  Translating {len(files)} file(s) to {state['translate_to']}")
    return {"_detail": f"{len(files)} file(s)"}


def translate_file(payload: dict, config) -> dict:
    from kb.translate.graph import translate_file as run_translation
    try:
        result = run_translation(payload["repo_path"], payload["file"], payload["to"],
                                 repo_name=payload["repo_name"], max_rounds=payload["rounds"],
                                 config=config)
        return {"translations": [{"file": payload["file"], **result}]}
    except Exception as e:
        return {"translations": [{"file": payload["file"], "error": str(e)}],
                "errors": [f"translate {payload['file']}: {e}"]}


STAGE_NODES = {"scan": scan, "parse": parse, "deps": deps, "extract": extract,
               "synthesize": synthesize, "embed": embed, "translate": translate_gate}


def _continue_or_stop(next_stage: str):
    def route(state: PipelineState) -> str:
        return END if state.get("errors") else next_stage
    return route


def build_ingest_graph(stages: list[str] = None, checkpointer=None):
    """A linear graph of the selected stages, in canonical order, where any
    failed stage routes straight to END."""
    stages = [s for s in STAGES if s in (stages or DEFAULT_STAGES)]
    if not stages:
        raise ValueError(f"no stages selected; choose from {STAGES}")
    g = StateGraph(PipelineState)
    for s in stages:
        g.add_node(s, STAGE_NODES[s])
    g.add_edge(START, stages[0])
    for a, b in zip(stages, stages[1:]):
        g.add_conditional_edges(a, _continue_or_stop(b), [b, END])
    last = stages[-1]
    if last == "translate":
        g.add_node("translate_file", translate_file)
        g.add_conditional_edges("translate", _translate_fan_out, ["translate_file", END])
        g.add_edge("translate_file", END)
    else:
        g.add_edge(last, END)
    return g.compile(checkpointer=checkpointer)


def run_pipeline(repo_path: str, stages: list[str] = None, repo_name: str = None,
                 config: dict | None = None, checkpointer=None, **options) -> dict:
    repo_root = Path(repo_path).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo path '{repo_path}' not found")
    state = {"repo_path": str(repo_root), "repo_name": repo_name or repo_root.name,
             "stage_log": [], "errors": [], "translations": [], **options}
    try:
        final = build_ingest_graph(stages, checkpointer).invoke(state, config or run_config())
    finally:
        _release_kuzu()
    _print_summary(final)
    return final


def _print_summary(state: PipelineState):
    print("\n══ summary " + "═" * 51)
    for s in state.get("stage_log", []):
        mark = "✓" if s["ok"] else "✗"
        print(f"  {mark} {s['stage']:<11} {s['seconds']:>7.1f}s  {s['detail']}")
    for t in state.get("translations", []):
        if t.get("error"):
            print(f"  ✗ translate {t['file']}: {t['error']}")
        else:
            rate = f"{t['pass_rate']:.0%}" if t.get("pass_rate") is not None else "unverified"
            print(f"  ✓ translate {t['file']}: {rate} → {t['output_dir']}")
    if state.get("errors"):
        print(f"\n  Stopped: {state['errors'][0]}")


def _parse_stages(only: str | None, skip: str | None) -> list[str]:
    stages = [s.strip() for s in only.split(",")] if only else list(DEFAULT_STAGES)
    if skip:
        stages = [s for s in stages if s not in {x.strip() for x in skip.split(",")}]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Choose from {', '.join(STAGES)}.")
    return stages


def main():
    p = argparse.ArgumentParser(description="Groundwork ingestion pipeline (LangGraph)")
    p.add_argument("repo_path")
    p.add_argument("--repo-name", default=None, help="Name stored in the DBs (default: folder name)")
    p.add_argument("--only", default=None, help=f"Comma-separated stages from: {', '.join(STAGES)}")
    p.add_argument("--skip", default=None, help="Comma-separated stages to leave out")
    p.add_argument("--clear", action="store_true", help="Clear this repo from Kùzu/Postgres first")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4, help="Max parallel LLM tasks")
    p.add_argument("--rate-limit", type=int, default=0, help="Max LLM calls per minute")
    p.add_argument("--lines", type=int, default=80, help="Lines per file sent for rule extraction")
    p.add_argument("--dep-lines", type=int, default=20, help="Lines per file scanned for imports")
    p.add_argument("--only-unprocessed", action="store_true")
    p.add_argument("--refine-rounds", type=int, default=0, help="Rule interrogation loop rounds")
    p.add_argument("--embed-method", choices=["fast", "bertscore"], default="fast")
    p.add_argument("--translate", nargs="*", default=[], metavar="FILE",
                   help="Files (relative to the repo) to translate; adds the translate stage")
    p.add_argument("--translate-to", default=None)
    p.add_argument("--translate-rounds", type=int, default=5)
    p.add_argument("--checkpoint", default=None,
                   help="SQLite file for LangGraph checkpoints (per-stage run history)")
    p.add_argument("--thread-id", default=None, help="Checkpoint thread (default: repo name)")
    args = p.parse_args()

    stages = _parse_stages(args.only, args.skip)
    if args.translate and "translate" not in stages:
        stages.append("translate")

    repo_name = args.repo_name or Path(args.repo_path).resolve().name
    checkpointer, cm = None, None
    if args.checkpoint:
        from langgraph.checkpoint.sqlite import SqliteSaver
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        cm = SqliteSaver.from_conn_string(args.checkpoint)
        checkpointer = cm.__enter__()
    config = run_config(args.model, args.provider, rate_limit=args.rate_limit,
                        max_concurrency=args.workers,
                        thread_id=args.thread_id or repo_name if checkpointer else None)
    try:
        final = run_pipeline(args.repo_path, stages, repo_name, config, checkpointer,
                             clear=args.clear, dep_lines=args.dep_lines, max_lines=args.lines,
                             only_unprocessed=args.only_unprocessed,
                             refine_rounds=args.refine_rounds, embed_method=args.embed_method,
                             translate_files=args.translate, translate_to=args.translate_to,
                             translate_rounds=args.translate_rounds)
    finally:
        if cm:
            cm.__exit__(None, None, None)
    sys.exit(1 if final.get("errors") else 0)


if __name__ == "__main__":
    main()
