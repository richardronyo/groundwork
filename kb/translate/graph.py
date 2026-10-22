"""
Groundwork — code translation as a LangGraph graph (AgentModernize workflow,
language-agnostic, reading from Groundwork's knowledge base).

    load ─► analyze ─► specify ─Send(node)─► node_tests (×N BSG nodes) ─► collect_tests
                                                                              │
                     ┌────────────────────────────────────────────────────────┘
                     ▼
                 translate ─► verify ─┬─ all pass / max rounds / same failures twice ─► finalize ─► END
                     ▲                │
                     └────────────────┘

  load         source file + KB context: the file's extracted business rules
               (Postgres), its parsed functions/classes (Postgres), and the
               files it imports (Kùzu DEPENDS_ON) with their stored BSGs
  analyze      Analyzer agent → RuleInventory, seeded with the KB rules so they
               are verified and extended rather than re-derived from scratch
  specify      Specification agent → BSG; cross-file depends_on edges point at
               dependency BSG nodes; the BSG is saved to Postgres (bsg_specs)
               so test generation and later translations can reuse it
  node_tests   Test agent, one Send per BSG node in parallel: language-neutral
               cases, run against the ORIGINAL code to record expected
               outcomes; cases the harness can't run are regenerated (≤3 tries)
  translate    Translation agent: line-by-line, BSG as a checklist; later
               rounds start from the best version so far plus the failures
  verify       runs every case against the translation, compares to the
               original's outcomes
  finalize     keeps the best-scoring round, writes code/BSG/tests/report,
               records the run in Postgres (translations)

Verification needs a harness for both languages (python, javascript, r — see
kb/verify/harness.py). For any other target the graph still translates, in a
single unverified round.

Usage:
    python3 -m kb.translate.graph repos/shop pricing.py --to javascript
    python3 -m kb.translate.graph repos/flask src/flask/helpers.py --to javascript --max-rounds 3
    python3 -m kb.translate.graph legacy/ calc.R --to python --no-kb
"""

from __future__ import annotations

import argparse
import json
import operator
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from kb.llm.client import llm_from_config, invoke_structured, run_config, DEFAULT_MODEL
from kb.agent.rag import db_key
from kb.graph.tree_to_json import get_language
from kb.verify.harness import (
    RUNNERS, runner_for, run_cases, compare_case, python_module_name,
)
from kb.translate.components import RuleInventory, BSG, TestSuite, Translation
from kb.translate import prompts as P

MAX_SOURCE_CHARS = 30_000
DEFAULT_MAX_ROUNDS = 5
DEFAULT_TEST_RETRIES = 3

# Extensions for targets without a verification harness (translated, unverified).
TARGET_EXTENSIONS = {
    "python": ".py", "javascript": ".js", "r": ".R", "typescript": ".ts", "java": ".java",
    "csharp": ".cs", "c#": ".cs", "go": ".go", "rust": ".rs", "kotlin": ".kt",
    "ruby": ".rb", "php": ".php", "cpp": ".cpp", "c++": ".cpp", "swift": ".swift",
    "scala": ".scala", "julia": ".jl",
}


class TranslationState(TypedDict, total=False):
    # inputs
    repo_root: str
    repo_name: str
    target_file: str
    target_language: str           # runner key or a language name
    max_rounds: int
    max_test_retries: int
    use_kb: bool
    out_dir: str | None
    # loaded
    db_key: str
    source_path: str
    source_language: str
    source_code: str
    module_name: str | None
    kb_rules: list[str]
    structure: dict | None
    dependencies: list[dict]       # [{"path", "bsg" | None, "rules": [...]}]
    # agents
    rules: dict                    # RuleInventory dump
    bsg: dict                      # BSG dump
    tests: Annotated[list[dict], operator.add]      # runnable cases + expected outcomes
    skipped: Annotated[list[dict], operator.add]    # cases that never ran on the source
    rounds: Annotated[list[dict], operator.add]     # one per translate→verify cycle
    round: int
    pending: dict | None           # translate → verify hand-off for the current round
    result: dict


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve(repo_root: Path, target_file: str) -> Path:
    from kb.agent.test_graph import resolve_target_path
    return resolve_target_path(repo_root, target_file)


def _target_key(lang: str) -> str:
    return lang.strip().lower()


def _target_ext(lang: str) -> str:
    key = _target_key(lang)
    if key in RUNNERS:
        return RUNNERS[key].extension
    return TARGET_EXTENSIONS.get(key, ".txt")


def _rules_text(rules: dict) -> str:
    return "\n".join(f"- [{r['category']}, {r['confidence']}] {r['description']}"
                     for r in rules.get("rules", [])) or "(none)"


def _node_text(node: dict) -> str:
    out = [f"{node['name']}  (functions: {', '.join(node['functions']) or '?'}; "
           f"confidence {node['confidence']})"]
    for label, key in (("inputs", "inputs"), ("outputs", "outputs")):
        if node[key]:
            out.append(f"  {label}: " + ", ".join(f"{f['name']}: {f['type']}" for f in node[key]))
    out += [f"  rule: {r['description']}" for r in node["rules"]]
    out += [f"  pre:  {p}" for p in node["preconditions"]]
    out += [f"  post: {p}" for p in node["postconditions"]]
    return "\n".join(out)


def _bsg_text(bsg: dict) -> str:
    parts = [_node_text(n) for n in bsg.get("nodes", [])]
    if bsg.get("edges"):
        parts.append("edges:\n" + "\n".join(
            f"  {e['source']} -[{e['label']}]-> {e['target']}" for e in bsg["edges"]))
    return "\n\n".join(parts)


def _structure_text(structure: dict | None) -> str:
    if not structure:
        return ""
    lines = []
    for f in structure.get("functions", []):
        lines.append(f"- def {f['name']}({', '.join(f['params'])})")
    for c in structure.get("classes", []):
        lines.append(f"- class {c['name']}" + (f"({', '.join(c['bases'])})" if c["bases"] else ""))
        for m in c.get("methods", []):
            lines.append(f"    - {m['name']}({', '.join(m['params'])})")
    return "\n".join(lines)


def _deps_text(deps: list[dict]) -> str:
    blocks = []
    for d in deps:
        if d.get("bsg"):
            nodes = "\n".join(f"  - {n['name']}: " + "; ".join(n["postconditions"][:3])
                              for n in d["bsg"].get("nodes", []))
            blocks.append(f"{d['path']} (BSG):\n{nodes}")
        elif d.get("rules"):
            blocks.append(f"{d['path']} (rules):\n" + "\n".join(f"  - {r}" for r in d["rules"][:8]))
    return "\n\n".join(blocks)


def _case_spec(case: dict) -> dict:
    """What the harness needs: the case without its recorded outcomes."""
    return {"id": case["id"], "steps": [{k: s[k] for k in ("id", "function", "args_json")}
                                        for s in case["steps"]]}


# ── nodes ─────────────────────────────────────────────────────────────────────

def load(state: TranslationState) -> dict:
    repo_root = Path(state["repo_root"]).resolve()
    path = _resolve(repo_root, state["target_file"])
    key = db_key(state["repo_name"], str(path.relative_to(repo_root)))
    language = get_language(path.name)
    code = path.read_text(errors="replace")
    if len(code) > MAX_SOURCE_CHARS:
        print(f"  ⚠️  {path.name} is {len(code):,} chars; translating the first {MAX_SOURCE_CHARS:,}.")
        code = code[:MAX_SOURCE_CHARS]

    kb_rules, structure, deps = [], None, []
    if state.get("use_kb", True):
        kb_rules, structure, deps = _load_kb(state["repo_name"], key, repo_root)
        print(f"  KB: {len(kb_rules)} stored rules, "
              f"{'parsed structure' if structure else 'no parsed structure'}, "
              f"{len(deps)} dependencies ({sum(1 for d in deps if d.get('bsg'))} with BSGs)")

    module_name = python_module_name(repo_root, path) if language == "Python" else None
    if not runner_for(language):
        print(f"  ⚠️  No test harness for {language}: tests can't record expected outcomes, "
              f"so the translation will be unverified.")
    return {"db_key": key, "source_path": str(path), "source_language": language,
            "source_code": code, "module_name": module_name, "kb_rules": kb_rules,
            "structure": structure, "dependencies": deps, "round": 0}


def _load_kb(repo_name: str, key: str, repo_root: Path):
    rules, structure, deps = [], None, []
    try:
        from kb.relationaldb.initialize_db import (
            get_connection, load_business_rules_from_db, load_structure_from_db, load_bsgs)
        conn = get_connection()
        try:
            all_rules = load_business_rules_from_db(conn, repo_name)
            rules = all_rules.get(key, [])
            structure = load_structure_from_db(conn, repo_name).get(key)
            dep_keys = _dependency_keys(repo_name, key)
            dep_bsgs = load_bsgs(conn, repo_name, dep_keys) if dep_keys else {}
            deps = [{"path": d, "bsg": dep_bsgs.get(d), "rules": all_rules.get(d, [])}
                    for d in dep_keys]
        finally:
            conn.close()
    except Exception as e:
        print(f"  (knowledge base unavailable, continuing from source only: {e})")
    return rules, structure, deps


def _dependency_keys(repo_name: str, key: str) -> list[str]:
    try:
        from kb.graph.kuzu_store import get_reader, get_dependencies
        return get_dependencies(get_reader(), [key], repo_name).get(key, [])
    except Exception:
        return []        # no graph yet (deps stage not run), or locked by a writer


def analyze(state: TranslationState, config) -> dict:
    kb = ""
    if state.get("kb_rules"):
        kb = P.KB_RULES_SECTION.format(rules="\n".join(f"- {r}" for r in state["kb_rules"]))
    name = Path(state["source_path"]).name
    prompt = P.ANALYZER_PROMPT.format(filename=name, language=state["source_language"],
                                      code=state["source_code"], kb_section=kb)
    inv = invoke_structured(llm_from_config(config, temperature=0.0), RuleInventory,
                            [("system", P.ANALYZER_SYSTEM), ("user", prompt)])
    print(f"  Analyzer: {len(inv.rules)} rules")
    return {"rules": inv.model_dump()}


def specify(state: TranslationState, config) -> dict:
    structure = _structure_text(state.get("structure"))
    deps = _deps_text(state.get("dependencies", []))
    prompt = P.SPECIFIER_PROMPT.format(
        filename=Path(state["source_path"]).name, language=state["source_language"],
        code=state["source_code"], rules=_rules_text(state["rules"]),
        structure_section=P.STRUCTURE_SECTION.format(structure=structure) if structure else "",
        deps_section=P.DEPS_SECTION.format(deps=deps) if deps else "")
    bsg = invoke_structured(llm_from_config(config, temperature=0.0), BSG,
                            [("system", P.SPECIFIER_SYSTEM), ("user", prompt)])
    bsg_dict = bsg.model_dump()
    cross = sum(1 for e in bsg.edges if e.label == "depends_on")
    print(f"  Specifier: {len(bsg.nodes)} nodes, {len(bsg.edges)} edges ({cross} cross-file)")
    if state.get("use_kb", True):
        _save_bsg(state["repo_name"], state["db_key"], bsg_dict, state["source_language"])
    return {"bsg": bsg_dict}


def _save_bsg(repo_name, key, bsg_dict, language):
    try:
        from kb.relationaldb.initialize_db import get_connection, get_file_id, save_bsg
        conn = get_connection()
        try:
            file_id = get_file_id(conn, repo_name, key)
            if file_id is None:
                return      # file not ingested; nothing to attach the BSG to
            save_bsg(conn, file_id, bsg_dict, language)
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"  (BSG not saved to the KB: {e})")


def fan_out_tests(state: TranslationState):
    nodes = state["bsg"].get("nodes", [])
    if not nodes or not runner_for(state["source_language"]):
        return "collect_tests"
    shared = {k: state[k] for k in ("repo_root", "source_path", "source_language",
                                    "source_code", "module_name")}
    shared["max_test_retries"] = state.get("max_test_retries", DEFAULT_TEST_RETRIES)
    return [Send("node_tests", {**shared, "node": n, "index": i}) for i, n in enumerate(nodes)]


def node_tests(payload: dict, config) -> dict:
    """Test agent for one BSG node. Every case is executed against the original
    code; its recorded outcome becomes the expected result. Cases the harness
    can't execute are fed back for regeneration, up to max_test_retries."""
    node, idx = payload["node"], payload["index"]
    runner = runner_for(payload["source_language"])
    llm = llm_from_config(config, temperature=0.4)
    kept: list[dict] = []
    problems: list[str] = []
    last_bad: list[dict] = []

    for attempt in range(1, payload["max_test_retries"] + 1):
        prefix = f"n{idx}a{attempt}_"
        feedback = P.TESTGEN_FEEDBACK.format(problems="\n".join(problems)) if problems else ""
        prompt = P.TESTGEN_PROMPT.format(
            filename=Path(payload["source_path"]).name, language=payload["source_language"],
            code=payload["source_code"], node=_node_text(node), n_unit=2,
            prefix=prefix, feedback=feedback)
        try:
            suite = invoke_structured(llm, TestSuite, [("system", P.TESTGEN_SYSTEM), ("user", prompt)])
        except Exception as e:
            problems = [f"generation failed: {e}"]
            continue

        cases = []
        for j, c in enumerate(suite.cases):
            d = c.model_dump()
            d["id"] = f"{prefix}{j}"          # ids unique across nodes and attempts
            d["node"] = node["name"]
            cases.append(d)
        if not cases:
            problems = ["no cases were produced"]
            continue

        res = run_cases(runner, Path(payload["source_path"]), [_case_spec(c) for c in cases],
                        Path(payload["repo_root"]), payload.get("module_name"))
        if "load_error" in res:
            # The source itself won't load in the harness: no retry can fix that.
            return {"skipped": [{**c, "reason": f"source failed to load: {res['load_error']}"}
                                for c in cases]}

        by_id = {r["id"]: r for r in res["results"]}
        problems, last_bad = [], []
        for c in cases:
            steps = by_id.get(c["id"], {}).get("steps", [])
            bad = next((s for s in steps if "harness_error" in s), None)
            if bad or len(steps) != len(c["steps"]):
                reason = bad["harness_error"] if bad else "not all steps ran"
                problems.append(f"- case {c['description']!r}: {reason}")
                last_bad.append({**c, "reason": reason})
                continue
            for spec_step, out in zip(c["steps"], steps):
                spec_step.update({k: out[k] for k in ("ok", "value", "error") if k in out})
            kept.append(c)
        if not problems:
            break

    print(f"  Tests for {node['name']}: {len(kept)} runnable"
          + (f", {len(last_bad)} skipped" if last_bad and problems else ""))
    return {"tests": kept, "skipped": last_bad if problems else []}


def collect_tests(state: TranslationState) -> dict:
    tests = state.get("tests", [])
    units = sum(1 for t in tests if t["kind"] == "unit")
    print(f"  Test suite: {len(tests)} cases ({units} unit, {len(tests) - units} behaviour), "
          f"{len(state.get('skipped', []))} skipped")
    return {}


def translate(state: TranslationState, config) -> dict:
    target = state["target_language"]
    key = _target_key(target)
    runner = RUNNERS.get(key)
    rnd = state.get("round", 0) + 1
    target_name = Path(state["source_path"]).stem + _target_ext(target)

    feedback = ""
    rounds = state.get("rounds", [])
    if rounds and rounds[-1].get("failures"):
        best = _best_round(rounds)
        last = rounds[-1]
        failures = "\n".join(f"- [{f['node']}] {f['description']}: {f['reason']}"
                             for f in last["failures"][:25])
        feedback = P.TRANSLATOR_FEEDBACK.format(
            round=last["round"], n_fail=len(last["failures"]), n_total=last["total"],
            failures=failures, best_round=best["round"], best_rate=best["pass_rate"] or 0,
            best_code=best["files"][0]["content"])

    system = P.TRANSLATOR_SYSTEM.format(
        source_language=state["source_language"], target_language=target,
        export_note=runner.export_note if runner else "")
    prompt = P.TRANSLATOR_PROMPT.format(
        filename=Path(state["source_path"]).name, source_language=state["source_language"],
        code=state["source_code"], bsg=_bsg_text(state["bsg"]), feedback=feedback,
        target_language=target, target_name=target_name)
    out = invoke_structured(llm_from_config(config, temperature=0.0), Translation,
                            [("system", system), ("user", prompt)])
    files = [f.model_dump() for f in out.files]
    # Make sure the main file is first and carries the expected name.
    files.sort(key=lambda f: Path(f["path"]).name != target_name)
    if files and Path(files[0]["path"]).name != target_name and len(files) == 1:
        files[0]["path"] = target_name
    for f in files:
        f["content"] = _strip_fences(f["content"])
    return {"round": rnd, "pending": {"round": rnd, "files": files, "notes": out.notes}}


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.rstrip() + "\n"


def verify(state: TranslationState) -> dict:
    pending = state["pending"]
    files = pending["files"]
    tests = state.get("tests", [])
    runner = RUNNERS.get(_target_key(state["target_language"]))
    entry = {"round": pending["round"], "files": files, "notes": pending["notes"],
             "total": len(tests), "passed": 0, "pass_rate": None, "failures": [],
             "failing": [], "verified": False}

    if not files:
        entry["failures"] = [{"id": "-", "node": "-", "description": "translation",
                              "reason": "no files returned"}]
    elif runner and runner.available() and tests:
        with tempfile.TemporaryDirectory(prefix="gw_translation_") as tmp:
            for f in files:
                p = Path(tmp) / Path(f["path"]).name
                p.write_text(f["content"])
            main = Path(tmp) / Path(files[0]["path"]).name
            res = run_cases(runner, main, [_case_spec(t) for t in tests])
        if "load_error" in res:
            reason = f"translation failed to load: {res['load_error'][:500]}"
            entry["failures"] = [{"id": t["id"], "node": t["node"], "description": t["description"],
                                  "reason": reason} for t in tests]
        else:
            by_id = {r["id"]: r for r in res["results"]}
            for t in tests:
                ok, reason = compare_case(t, by_id.get(t["id"]))
                if ok:
                    entry["passed"] += 1
                else:
                    entry["failures"].append({"id": t["id"], "node": t["node"],
                                              "description": t["description"], "reason": reason})
        entry["verified"] = True
        entry["pass_rate"] = entry["passed"] / len(tests)
    entry["failing"] = sorted(f["id"] for f in entry["failures"])
    rate = f"{entry['pass_rate']:.0%}" if entry["pass_rate"] is not None else "unverified"
    print(f"  Round {entry['round']}: {entry['passed']}/{entry['total']} cases agree ({rate})")
    return {"rounds": [entry], "pending": None}


def _best_round(rounds: list[dict]) -> dict:
    # Highest pass rate; earliest round on ties (later rounds can regress).
    return max(rounds, key=lambda r: ((r["pass_rate"] or 0), -r["round"]))


def after_verify(state: TranslationState) -> str:
    rounds = state["rounds"]
    last = rounds[-1]
    if not last["verified"]:
        return "finalize"
    if last["pass_rate"] == 1.0:
        return "finalize"
    if last["round"] >= state.get("max_rounds", DEFAULT_MAX_ROUNDS):
        return "finalize"
    if len(rounds) >= 2 and rounds[-2]["failing"] == last["failing"]:
        return "finalize"          # same failing tests twice in a row: feedback isn't helping
    return "translate"


def finalize(state: TranslationState) -> dict:
    rounds = state["rounds"]
    best = _best_round(rounds)
    target = state["target_language"]
    repo_root = Path(state["repo_root"])
    out_dir = Path(state.get("out_dir") or
                   Path("outputs") / "translations" / state["repo_name"] /
                   f"{Path(state['source_path']).stem}_{_target_key(target)}")
    _write_outputs(out_dir, state, best)

    result = {"best_round": best["round"], "rounds": len(rounds), "pass_rate": best["pass_rate"],
              "n_tests": len(state.get("tests", [])), "output_dir": str(out_dir),
              "main_file": str(out_dir / Path(best["files"][0]["path"]).name) if best["files"] else None}
    if state.get("use_kb", True):
        try:
            from kb.relationaldb.initialize_db import get_connection, save_translation_run
            conn = get_connection()
            try:
                save_translation_run(conn, state["repo_name"], state["db_key"],
                                     state["source_language"], target, len(rounds),
                                     best["round"], best["pass_rate"], result["n_tests"],
                                     str(out_dir))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f"  (translation run not recorded in the KB: {e})")

    rate = f"{best['pass_rate']:.0%}" if best["pass_rate"] is not None else "unverified"
    print(f"\n  ✓ Kept round {best['round']} of {len(rounds)} ({rate}) → {out_dir}")
    return {"result": result}


def _write_outputs(out_dir: Path, state: TranslationState, best: dict):
    tests_dir = out_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for f in best["files"]:
        (out_dir / Path(f["path"]).name).write_text(f["content"])
    (out_dir / "rules.json").write_text(json.dumps(state["rules"], indent=2))
    (out_dir / "bsg.json").write_text(json.dumps(state["bsg"], indent=2))

    tests, skipped = state.get("tests", []), state.get("skipped", [])
    failing = set(best["failing"])
    (tests_dir / "tests.json").write_text(json.dumps(
        {"tests": tests, "skipped": skipped, "best_round_failing": sorted(failing)}, indent=2))

    def case_md(c):
        status = "FAIL" if c["id"] in failing else "pass"
        lines = [f"### {c['id']} — {c['description']}  [{status}]", f"Node: {c['node']}", ""]
        for s in c["steps"]:
            outcome = (f"→ {json.dumps(s.get('value'))}" if s.get("ok")
                       else f"→ error: {s.get('error')}" if "ok" in s else "")
            lines.append(f"- `{s['id']}`: `{s['function']}({s['args_json'][1:-1]})` {outcome}")
        return "\n".join(lines) + "\n"

    for kind, fname in (("unit", "unit_tests.md"), ("behaviour", "behaviour_tests.md")):
        sel = [c for c in tests if c["kind"] == kind]
        (tests_dir / fname).write_text(f"# {kind.title()} tests ({len(sel)})\n\n"
                                       + "\n".join(case_md(c) for c in sel))
    (tests_dir / "skipped_tests.md").write_text(
        f"# Skipped tests ({len(skipped)})\n\nThese never ran against the source.\n\n"
        + "\n".join(f"- {c['id']} [{c['node']}] {c['description']}: {c['reason']}" for c in skipped))

    lines = [f"# Translation report: {Path(state['source_path']).name} → {state['target_language']}", "",
             f"- Source language: {state['source_language']}",
             f"- KB rules used: {len(state.get('kb_rules', []))}; dependencies: "
             f"{len(state.get('dependencies', []))}",
             f"- Rules: {len(state['rules'].get('rules', []))}; BSG nodes: "
             f"{len(state['bsg'].get('nodes', []))}",
             f"- Test cases: {len(tests)} ({len(skipped)} skipped)",
             f"- Kept round: {best['round']}", "",
             "| Round | Agree | Pass rate |", "|---|---|---|"]
    for r in state["rounds"]:
        rate = f"{r['pass_rate']:.0%}" if r["pass_rate"] is not None else "unverified"
        lines.append(f"| {r['round']} | {r['passed']}/{r['total']} | {rate} |")
    if best.get("notes"):
        lines += ["", "## Translator notes", "", best["notes"]]
    if best["failures"]:
        lines += ["", f"## Still failing in round {best['round']}", ""]
        lines += [f"- {f['id']} [{f['node']}] {f['description']}: {f['reason']}" for f in best["failures"]]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    for r in state["rounds"]:
        (tests_dir / f"round_{r['round']}.json").write_text(json.dumps(
            {k: r[k] for k in ("round", "passed", "total", "pass_rate", "failures")}, indent=2))


# ── graph ─────────────────────────────────────────────────────────────────────

def build_translation_graph(checkpointer=None):
    g = StateGraph(TranslationState)
    g.add_node("load", load)
    g.add_node("analyze", analyze)
    g.add_node("specify", specify)
    g.add_node("node_tests", node_tests)
    g.add_node("collect_tests", collect_tests)
    g.add_node("translate", translate)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_edge(START, "load")
    g.add_edge("load", "analyze")
    g.add_edge("analyze", "specify")
    g.add_conditional_edges("specify", fan_out_tests, ["node_tests", "collect_tests"])
    g.add_edge("node_tests", "collect_tests")
    g.add_edge("collect_tests", "translate")
    g.add_edge("translate", "verify")
    g.add_conditional_edges("verify", after_verify, ["translate", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def translate_file(repo_root: str | Path, target_file: str, target_language: str,
                   repo_name: str | None = None, max_rounds: int = DEFAULT_MAX_ROUNDS,
                   use_kb: bool = True, out_dir: str | None = None,
                   max_test_retries: int = DEFAULT_TEST_RETRIES,
                   config: dict | None = None) -> dict:
    repo_root = Path(repo_root).resolve()
    state = build_translation_graph().invoke({
        "repo_root": str(repo_root), "repo_name": repo_name or repo_root.name,
        "target_file": target_file, "target_language": target_language,
        "max_rounds": max_rounds, "max_test_retries": max_test_retries,
        "use_kb": use_kb, "out_dir": out_dir,
        "tests": [], "skipped": [], "rounds": [],
    }, config or run_config())
    return state["result"]


def main():
    p = argparse.ArgumentParser(description="Groundwork — translate a file (LangGraph)")
    p.add_argument("repo", help="Path to the repository")
    p.add_argument("target_file", help="File to translate, relative to the repo root")
    p.add_argument("--to", required=True, dest="target_language",
                   help=f"Target language (verified: {', '.join(RUNNERS)}; others unverified)")
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--test-retries", type=int, default=DEFAULT_TEST_RETRIES)
    p.add_argument("--no-kb", action="store_true", help="Ignore the knowledge base (source only)")
    p.add_argument("--out", default=None, help="Output folder")
    p.add_argument("--workers", type=int, default=4, help="BSG nodes tested in parallel")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    result = translate_file(args.repo, args.target_file, args.target_language,
                            max_rounds=args.max_rounds, use_kb=not args.no_kb, out_dir=args.out,
                            max_test_retries=args.test_retries,
                            config=run_config(args.model, args.provider,
                                              max_concurrency=args.workers))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
