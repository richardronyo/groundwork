"""
Groundwork — test generation as LangGraph graphs.

Both modes share every node; only the edges differ, which is what makes the
paper's single-pass vs agentic ablation a clean comparison:

  single:   load_context → retrieve → generate → remember → END

  agentic:  load_context → retrieve → generate → run_tests ─┬─ passed ───────────► remember → END
                                          ▲                  ├─ out of attempts ───► remember
                                          └──── reflect ◄────┘

  load_context  source file, prior memories for this file, and (if the
                translation graph has built one) the file's stored BSG
  retrieve      RAG: business rules, top key point, dependency sources
                (skipped unless use_rag)
  generate      one structured LLM call → GeneratedTest
  run_tests     runs the test with pytest against the repo
  reflect       structured diagnosis (TestReflection) fed into the next generate;
                a "code" fault means the test found a real bug, so the next
                attempt marks it xfail(strict=True) instead of weakening it
  remember      stores the exact prompt/response, plus reflections, in agent memory

Every memory read and write uses the repo-prefixed DB key, the same key RAG,
business_rules, Chroma and Kùzu use.
"""

from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from kb.llm.client import llm_from_config, invoke_structured
from kb.llm.schemas import GeneratedTest, TestReflection
from kb.agent.rag import db_key, gather_rag_context, format_rag_context
from kb.verify.pytest_runner import run_pytest

MAX_READ_CHARS = 6000

SYSTEM_PROMPT = """You are a test-generation assistant for a Python codebase using pytest.
You will be given a source file's contents and asked to write a pytest test file for it.
Cover the normal case plus at least one edge case. Import the code under test the
way a user of the package would. Do not mock the function under test itself."""

XFAIL_NOTE = """If a test exposes a genuine bug in the source (the code contradicts
its documented or specified behaviour), keep the test and mark it
@pytest.mark.xfail(strict=True, reason="<the bug>") instead of weakening it."""


class TestGenState(TypedDict, total=False):
    # inputs
    repo_root: str
    repo_name: str
    target_file: str            # relative to repo_root, as typed on the CLI
    function_name: str | None
    mode: Literal["single", "agentic"]
    use_rag: bool
    use_memory: bool
    use_bsg: bool
    top_n_rules: int
    top_n_deps: int
    max_attempts: int
    # context
    db_key: str
    source: str
    prior_notes: list[str]
    bsg: dict | None
    rag_text: str
    # working state
    prompt: str
    response: str
    test_code: str
    run: dict | None
    attempt: int
    reflection: dict | None
    attempts: Annotated[list[dict], operator.add]     # one per generate→run cycle


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_target_path(repo_root: Path, target_file: str) -> Path:
    """Accepts target_file with or without a leading '<repo_name>/' — the
    doubled-prefix tolerance from the old generate_test.py."""
    direct = (repo_root / target_file).resolve()
    if direct.exists():
        return direct
    prefix = f"{repo_root.name}/"
    if target_file.startswith(prefix):
        stripped = (repo_root / target_file[len(prefix):]).resolve()
        if stripped.exists():
            return stripped
    raise FileNotFoundError(
        f"'{target_file}' not found under {repo_root} — target_file should be relative "
        f"to the repo root, without the repo name repeated.")


def _strip_fences(code: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
    return (m.group(1) if m else code).strip() + "\n"


def _format_bsg(bsg: dict) -> str:
    lines = []
    for n in bsg.get("nodes", []):
        lines.append(f"- {n['name']} (confidence {n.get('confidence', '?')})")
        for pre in n.get("preconditions", []):
            lines.append(f"    pre:  {pre}")
        for post in n.get("postconditions", []):
            lines.append(f"    post: {post}")
    return "\n".join(lines)


# ── nodes ─────────────────────────────────────────────────────────────────────

def load_context(state: TestGenState) -> dict:
    repo_root = Path(state["repo_root"])
    key = db_key(state["repo_name"], state["target_file"])
    source = resolve_target_path(repo_root, state["target_file"]).read_text(
        errors="replace")[:MAX_READ_CHARS]

    notes: list[str] = []
    if state.get("use_memory", True):
        try:
            from kb.agent.memory import recall_by_type
            # Reflections and corrections are the useful carry-over; raw
            # past prompts would just bloat this prompt.
            notes = (recall_by_type(state["repo_name"], key, "reflection", 3)
                     + recall_by_type(state["repo_name"], key, "correction", 3))
        except Exception as e:
            print(f"  (memory unavailable: {e})")

    bsg = None
    if state.get("use_bsg", True):
        try:
            from kb.relationaldb.initialize_db import get_connection, load_bsgs
            conn = get_connection()
            try:
                bsg = load_bsgs(conn, state["repo_name"], [key]).get(key)
            finally:
                conn.close()
        except Exception:
            bsg = None

    return {"db_key": key, "source": source, "prior_notes": notes, "bsg": bsg,
            "attempt": 0, "run": None, "reflection": None}


def retrieve(state: TestGenState) -> dict:
    if not state.get("use_rag"):
        return {"rag_text": ""}
    ctx = gather_rag_context(Path(state["repo_root"]), state["repo_name"], state["target_file"],
                             top_n_rules=state.get("top_n_rules", 5),
                             top_n_deps=state.get("top_n_deps", 3))
    text = format_rag_context(ctx)
    if not text:
        print("  (--rag requested, but nothing retrieved for this file — "
              "have extract/embeddings/deps been run?)")
    return {"rag_text": text}


def build_prompt(state: TestGenState) -> str:
    task = f"Source file `{state['target_file']}`"
    if state.get("function_name"):
        task += f" (write tests specifically for `{state['function_name']}`)"
    task += f":\n\n```python\n{state['source']}\n```"
    if state.get("rag_text"):
        task += "\n\n" + state["rag_text"]
    if state.get("bsg"):
        task += ("\n\nBehavioural specification for this file (test these pre/postconditions):\n"
                 + _format_bsg(state["bsg"]))
    if state.get("prior_notes"):
        task += "\n\nNotes from previous work on this file:\n" + "\n".join(
            f"- {n}" for n in state["prior_notes"])
    if state.get("reflection") and state.get("test_code"):
        r = state["reflection"]
        task += (f"\n\nYour previous attempt failed.\n\n```python\n{state['test_code']}```\n\n"
                 f"pytest output (tail):\n```\n{state['run']['output'][-2000:]}\n```\n\n"
                 f"Diagnosis ({r['fault']} fault): {r['cause']}\nFixes:\n"
                 + "\n".join(f"- {f}" for f in r["fixes"])
                 + "\n\nWrite the complete corrected test file.")
    return task


def generate(state: TestGenState, config) -> dict:
    system = SYSTEM_PROMPT + ("\n\n" + XFAIL_NOTE if state.get("mode") == "agentic" else "")
    prompt = build_prompt(state)
    out = invoke_structured(llm_from_config(config, temperature=0.2), GeneratedTest,
                            [("system", system), ("user", prompt)])
    return {"prompt": prompt, "response": out.model_dump_json(),
            "test_code": _strip_fences(out.code), "attempt": state.get("attempt", 0) + 1}


def run_tests(state: TestGenState) -> dict:
    result = run_pytest(state["test_code"], state["repo_root"],
                        filename=f"test_{Path(state['target_file']).stem}.py")
    mark = "✓" if result["passed"] else "✗"
    print(f"  attempt {state['attempt']}: {mark} {result['counts'] or result['exit_code']}")
    return {"run": result,
            "attempts": [{"attempt": state["attempt"], "passed": result["passed"],
                          "counts": result["counts"], "code": state["test_code"]}]}


def reflect(state: TestGenState, config) -> dict:
    prompt = (f"This pytest file was generated for `{state['target_file']}`:\n\n"
              f"```python\n{state['test_code']}```\n\nSource under test:\n\n"
              f"```python\n{state['source']}\n```\n\nRunning it produced:\n\n"
              f"```\n{state['run']['output']}\n```\n\nDiagnose the failure.")
    r = invoke_structured(llm_from_config(config, temperature=0.0), TestReflection,
                          [("system", "You debug failing generated tests."), ("user", prompt)])
    return {"reflection": r.model_dump()}


def remember_node(state: TestGenState) -> dict:
    if not state.get("use_memory", True):
        return {}
    try:
        from kb.agent.memory import remember, remember_interaction
        remember_interaction(state["repo_name"], state["db_key"],
                             input_text=state["prompt"], output_text=state["response"])
        if state.get("reflection"):
            r = state["reflection"]
            remember(state["repo_name"],
                     f"Test generation ({r['fault']} fault): {r['cause']} "
                     f"Fixes: {'; '.join(r['fixes'])}",
                     memory_type="reflection", file_path=state["db_key"])
    except Exception as e:
        print(f"  (could not store memory: {e})")
    return {}


# ── routing ───────────────────────────────────────────────────────────────────

def _after_run(state: TestGenState) -> str:
    if state["run"]["passed"] or state["attempt"] >= state.get("max_attempts", 3):
        return "remember"
    return "reflect"


def build_test_graph(mode: Literal["single", "agentic"] = "single", checkpointer=None):
    """mode only changes edges: the agentic graph adds run_tests and the
    reflect loop between generate and remember."""
    g = StateGraph(TestGenState)
    g.add_node("load_context", load_context)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("remember", remember_node)
    g.add_edge(START, "load_context")
    g.add_edge("load_context", "retrieve")
    g.add_edge("retrieve", "generate")
    if mode == "agentic":
        g.add_node("run_tests", run_tests)
        g.add_node("reflect", reflect)
        g.add_edge("generate", "run_tests")
        g.add_conditional_edges("run_tests", _after_run, ["reflect", "remember"])
        g.add_edge("reflect", "generate")
    else:
        g.add_edge("generate", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=checkpointer)
