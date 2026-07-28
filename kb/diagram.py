#!/usr/bin/env python3
"""
Groundwork — UML Diagram Generator

Renders relationships from the knowledge base as Mermaid UML class diagrams
(which GitHub, Jupyter, and most Markdown viewers render natively).

Five diagram types:

  deps      Which files depend on a file, and which it depends on.
            Source: Kùzu DEPENDS_ON edges (both directions).

  function  Which files define and call a given function.
            Source: AST scan of the repo source. The graph is FILE-level —
            it has no Function nodes or CALLS edges — so this reads the
            source that the knowledge base indexes.

  class     True UML class diagram — attributes, methods, and inheritance /
            composition. Give --class <Name> for one class's hierarchy
            (ancestors, descendants, composed types), or --file <path> for
            every class defined in a file. Source: AST scan (Python, exact)
            or regex scan (C#, JS/TS, approximate) of the repo source.

  keypoint  Which files align most closely to a repo key point.
            Source: ChromaDB vectors (dimension k = similarity to key point k).

  system    Whole-repo overview aggregated to module level (configurable
            path depth): module-to-module dependency edges plus per-module
            file/class counts. Source: Kùzu DEPENDS_ON edges + class scan.

Run from the PROJECT ROOT.

Usage:
    python3 -m kb.diagram --repo flask --type deps --file src/flask/app.py
    python3 -m kb.diagram --repo flask --type function --function make_response
    python3 -m kb.diagram --repo flask --type class --class Flask
    python3 -m kb.diagram --repo flask --type class --file src/flask/app.py
    python3 -m kb.diagram --repo flask --type keypoint --kp 3
    python3 -m kb.diagram --repo flask --type keypoint --kp 3 --out diagrams/kp3.md
    python3 -m kb.diagram --repo flask --type system

Requirements:
    pip install kuzu chromadb psycopg python-dotenv networkx matplotlib
"""

import os
import re
import ast
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH    = "./chroma_db"
CHROMA_COLLECTION = "groundwork"


# ── Helpers ───────────────────────────────────────────────────────────────────

def node_id(path: str) -> str:
    """Mermaid-safe identifier from a file path."""
    ident = re.sub(r"[^0-9a-zA-Z]", "_", path)
    if ident and ident[0].isdigit():
        ident = "f_" + ident
    return ident or "root"


def mermaid_safe(text: str, limit: int = 70) -> str:
    """
    Makes arbitrary text safe to place inside a Mermaid class body.

    Key points are LLM-generated prose, so they can contain newlines, quotes,
    braces, and HTML — all of which break Mermaid's parser. Collapse to a
    single line and neutralise the metacharacters.
    """
    if not text:
        return ""
    text = " ".join(str(text).split())          # collapse newlines/runs of space
    text = text.replace('"', "'")                # quotes end a label
    text = text.replace("{", "(").replace("}", ")")   # braces end a class body
    text = text.replace("<", "&lt;").replace(">", "&gt;")  # HTML is interpreted
    text = text.replace("|", "/")                # pipes appear in link syntax
    if len(text) > limit:
        text = text[:limit - 1] + "…"
    return text


def comment_safe(text: str) -> str:
    """One-line, safe for a %% Mermaid comment."""
    return " ".join(str(text or "").split())


def short(path: str, keep: int = 2) -> str:
    """Trailing path segments, for readable labels."""
    parts = Path(path).parts
    return "/".join(parts[-keep:]) if len(parts) > keep else path


def collection_name_for(repo_name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in (repo_name or "").lower())
    return f"{CHROMA_COLLECTION}_{safe}"


def resolve_repo_path(repo_name: str, explicit: str = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            sys.exit(f"Error: --repo-path not found: {p}")
        return p
    for cand in (Path("./repos") / repo_name, Path(repo_name)):
        if cand.is_dir():
            return cand
    sys.exit(f"Error: repo not found on disk. Pass --repo-path. Tried ./repos/{repo_name}, ./{repo_name}")


def try_resolve_repo_path(repo_name: str, explicit: str = None):
    """Like resolve_repo_path, but returns None instead of exiting — used by
    --type system, which can still draw a module dependency graph from Kùzu
    alone even when the source isn't on disk (just without class counts)."""
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    for cand in (Path("./repos") / repo_name, Path(repo_name)):
        if cand.is_dir():
            return cand
    return None


def module_of(path: str, depth: int = 2) -> str:
    """The module a file belongs to: its first `depth` path segments."""
    parts = Path(path).parts
    if len(parts) <= 1:
        return "(root)"
    return "/".join(parts[:min(depth, len(parts) - 1)])


# ── 1. Dependency diagram (from Kùzu) ─────────────────────────────────────────

def deps_data(repo_name: str, rel_file: str, depth: int = 1) -> dict:
    """
    Gathers a file's dependency neighbourhood once, so the Mermaid emitter and
    the image renderer draw from identical data instead of querying twice.
    """
    from kb.graph.kuzu_store import get_connection, rows_to_dicts

    conn = get_connection()

    out = [r[0] for r in rows_to_dicts(conn.execute(f"""
        MATCH (f:File {{repository_name: $repo, relative: $rel}})
              -[:DEPENDS_ON*1..{depth}]->(d:File)
        RETURN DISTINCT d.relative ORDER BY d.relative
    """, {"repo": repo_name, "rel": rel_file}))]

    inc = [r[0] for r in rows_to_dicts(conn.execute(f"""
        MATCH (s:File {{repository_name: $repo}})
              -[:DEPENDS_ON*1..{depth}]->(f:File {{relative: $rel}})
        RETURN DISTINCT s.relative ORDER BY s.relative
    """, {"repo": repo_name, "rel": rel_file}))]

    direct_out = [r[0] for r in rows_to_dicts(conn.execute("""
        MATCH (f:File {repository_name: $repo, relative: $rel})-[:DEPENDS_ON]->(d:File)
        RETURN DISTINCT d.relative
    """, {"repo": repo_name, "rel": rel_file}))]

    direct_in = [r[0] for r in rows_to_dicts(conn.execute("""
        MATCH (s:File {repository_name: $repo})-[:DEPENDS_ON]->(f:File {relative: $rel})
        RETURN DISTINCT s.relative
    """, {"repo": repo_name, "rel": rel_file}))]

    return {"target": rel_file, "out": out, "inc": inc,
            "direct_out": direct_out, "direct_in": direct_in,
            "metrics": _get_metrics(repo_name, [rel_file] + out + inc)}


def diagram_deps(repo_name: str, rel_file: str, depth: int = 1) -> str:
    """
    UML class diagram of a file's dependencies (outgoing) and dependents
    (incoming). Both directions come straight from the Kùzu DEPENDS_ON edges.
    """
    _d = deps_data(repo_name, rel_file, depth)
    out, inc = _d["out"], _d["inc"]
    direct_out, direct_in = _d["direct_out"], _d["direct_in"]

    if not out and not inc:
        return (f"%% No DEPENDS_ON edges for {rel_file} in '{repo_name}'.\n"
                f"%% Run the deps stage, or check the path.\n"
                f"classDiagram\n    class {node_id(rel_file)}[\"{short(rel_file)}\"]\n")

    metrics = _d["metrics"]

    lines = ["classDiagram"]
    lines.append(f"    direction LR")

    def emit_class(path, stereotype=None):
        nid = node_id(path)
        m = metrics.get(path)
        lines.append(f'    class {nid}["{mermaid_safe(short(path))}"] {{')
        if stereotype:
            lines.append(f"        <<{stereotype}>>")
        if m:
            if m["classes"]:
                lines.append(f"        +{m['classes']} classes")
            if m["functions"]:
                lines.append(f"        +{m['functions']} functions")
            if m["methods"]:
                lines.append(f"        +{m['methods']} methods")
            lines.append(f"        +{m['lines']} lines")
        lines.append("    }")

    emit_class(rel_file, "target")
    for p in inc:
        emit_class(p)
    for p in out:
        emit_class(p)

    # UML dependency arrows: ..> means "depends on"
    for p in direct_in:
        lines.append(f"    {node_id(p)} ..> {node_id(rel_file)} : imports")
    for p in direct_out:
        lines.append(f"    {node_id(rel_file)} ..> {node_id(p)} : imports")

    header = (f"%% Dependencies for {rel_file} ({repo_name})\n"
              f"%% {len(direct_in)} direct dependents, {len(direct_out)} direct dependencies\n")
    return header + "\n".join(lines) + "\n"


def _get_metrics(repo_name: str, paths: list) -> dict:
    """Per-file metrics from PostgreSQL, for populating the UML classes."""
    if not paths:
        return {}
    try:
        import psycopg
        from kb.relationaldb.initialize_db import get_connection as pg_conn
        conn = pg_conn()
    except Exception:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT file_path, classes, functions, methods, lines
                FROM files
                WHERE repository_name = %s AND file_path = ANY(%s)
            """, (repo_name, list(set(paths))))
            return {r[0]: {"classes": r[1], "functions": r[2],
                           "methods": r[3], "lines": r[4]}
                    for r in cur.fetchall()}
    except Exception:
        return {}
    finally:
        conn.close()


# ── 2. Function diagram (AST scan — the graph has no CALLS edges) ─────────────

# Languages we can scan for function definitions / call sites.
# Python gets a real AST parse; the rest use regex, which is approximate but
# honest — C#/Razor/JS have no stdlib parser available here.
SCANNABLE = {
    ".py":     "Python",
    ".cs":     "C#",
    ".cshtml": "Razor",
    ".razor":  "Razor",
    ".js":     "JavaScript",
    ".jsx":    "JavaScript",
    ".ts":     "TypeScript",
    ".tsx":    "TypeScript",
}


def _iter_source_files(repo_path: Path, repo_name: str, languages=None):
    """
    Yields (relative, absolute) for files the graph knows about, optionally
    filtered to specific languages. Falls back to walking the repo.
    """
    rels = []
    try:
        from kb.graph.kuzu_store import get_connection, rows_to_dicts
        conn = get_connection()
        if languages:
            rows = rows_to_dicts(conn.execute("""
                MATCH (f:File {repository_name: $repo})
                WHERE list_contains($langs, f.language)
                RETURN f.relative ORDER BY f.relative
            """, {"repo": repo_name, "langs": list(languages)}))
        else:
            rows = rows_to_dicts(conn.execute("""
                MATCH (f:File {repository_name: $repo})
                RETURN f.relative ORDER BY f.relative
            """, {"repo": repo_name}))
        rels = [r[0] for r in rows]
    except Exception:
        rels = []

    if not rels:  # graph empty — walk the repo instead
        wanted_exts = ([ext for ext, lang in SCANNABLE.items() if lang in languages]
                       if languages else SCANNABLE.keys())
        for ext in wanted_exts:
            rels += [str(p.relative_to(repo_path)) for p in repo_path.rglob(f"*{ext}")]

    for rel in rels:
        if Path(rel).suffix.lower() not in SCANNABLE:
            continue
        full = repo_path / rel
        if full.is_file():
            yield rel, full


def _scan_python(text, func_name):
    """Exact scan via AST. Returns (defines, imports, call_count)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return False, False, 0
    defines = imports = False
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                defines = True
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == func_name for a in node.names):
                imports = True
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                   fn.attr if isinstance(fn, ast.Attribute) else None)
            if name == func_name:
                calls += 1
    return defines, imports, calls


def _strip_comments(text, lang):
    """Removes comments/strings so call counts aren't inflated by prose."""
    if lang in ("C#", "JavaScript", "TypeScript", "Razor"):
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)   # block comments
        text = re.sub(r"//[^\n]*", " ", text)                # line comments
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)      # string literals
    return text


C_SHARP_DEF = (
    r"(?:public|private|protected|internal|static|virtual|override|async|sealed|partial|\s)+"
    r"[\w<>\[\],\.\?]+\s+{name}\s*(?:<[^>]*>)?\s*\("
)
JS_DEF_PATTERNS = [
    r"function\s+{name}\s*\(",                    # function foo(
    r"(?:const|let|var)\s+{name}\s*=\s*(?:async\s*)?(?:function)?\s*\(",  # const foo = (
    r"(?:const|let|var)\s+{name}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",     # const foo = () =>
    r"{name}\s*:\s*(?:async\s*)?function\s*\(",  # foo: function(
    r"{name}\s*\([^)]*\)\s*\{{",                 # foo() {  (method shorthand)
]


def _scan_regex(text, func_name, lang):
    """Approximate scan for brace languages. Returns (defines, imports, calls)."""
    text = _strip_comments(text, lang)
    esc = re.escape(func_name)

    defines = False
    if lang in ("C#", "Razor"):
        if re.search(C_SHARP_DEF.format(name=esc), text):
            defines = True
    if lang in ("JavaScript", "TypeScript", "Razor"):
        for pat in JS_DEF_PATTERNS:
            if re.search(pat.format(name=esc), text):
                defines = True
                break

    # C# `using X;` / JS `import { foo } from` — approximate "brought into scope"
    imports = bool(re.search(r"import\s*\{{[^}}]*\b{n}\b[^}}]*\}}".format(n=esc), text)) or \
              bool(re.search(r"import\s+{n}\s+from".format(n=esc), text))

    # Call sites: foo(  or  obj.foo(  — excluding the definition lines
    calls = len(re.findall(r"(?<![\w.]){n}\s*\(".format(n=esc), text)) + \
            len(re.findall(r"\.{n}\s*\(".format(n=esc), text))
    if defines:
        calls = max(calls - 1, 0)   # don't count the declaration itself
    return defines, imports, calls


def scan_function(repo_path: Path, repo_name: str, func_name: str, languages=None):
    """
    Finds every file that DEFINES, IMPORTS, or CALLS `func_name`.

    Python is parsed with the AST (exact). C#, Razor, and JS/TS are matched with
    regex (approximate) because the knowledge base models dependencies at FILE
    level — it has no Function nodes or CALLS edges to query.

    Returns (definers, importers, callers{rel: count}).
    """
    definers, importers, callers = [], [], defaultdict(int)

    for rel, full in _iter_source_files(repo_path, repo_name, languages):
        ext = Path(rel).suffix.lower()
        lang = SCANNABLE.get(ext)
        try:
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if func_name not in text:      # cheap prefilter — most files skip here
            continue

        if lang == "Python":
            d, i, c = _scan_python(text, func_name)
        else:
            d, i, c = _scan_regex(text, func_name, lang)

        if d and rel not in definers:
            definers.append(rel)
        if i and rel not in importers:
            importers.append(rel)
        if c:
            callers[rel] += c

    return definers, importers, dict(callers)


def diagram_function(repo_name: str, repo_path: Path, func_name: str,
                     max_callers: int = 25, languages=None) -> str:
    definers, importers, callers = scan_function(repo_path, repo_name, func_name, languages)

    if not definers and not callers and not importers:
        return (f"%% '{func_name}' not found in any scanned source file of '{repo_name}'.\n"
                f"classDiagram\n    class missing[\"{func_name} — not found\"]\n")

    lines = ["classDiagram", "    direction LR"]

    # The function itself, as a UML interface-style node
    fid = f"fn_{node_id(func_name)}"
    lines.append(f'    class {fid}["{mermaid_safe(func_name)}()"] {{')
    lines.append("        <<function>>")
    if definers:
        lines.append(f"        defined in {len(definers)} file(s)")
    lines.append(f"        called in {len(callers)} file(s)")
    lines.append("    }")

    for d in definers:
        nid = node_id(d)
        lines.append(f'    class {nid}["{mermaid_safe(short(d))}"] {{')
        lines.append("        <<defines>>")
        lines.append("    }")
        # UML realization: this file provides the function
        lines.append(f"    {nid} --|> {fid} : defines")

    # Rank callers by call count; cap so the diagram stays readable
    ranked = sorted(callers.items(), key=lambda x: -x[1])
    shown = ranked[:max_callers]
    for rel, count in shown:
        if rel in definers:
            continue
        nid = node_id(rel)
        stereo = "imports+calls" if rel in importers else "calls"
        lines.append(f'    class {nid}["{mermaid_safe(short(rel))}"] {{')
        lines.append(f"        <<{stereo}>>")
        lines.append(f"        {count} call site(s)")
        lines.append("    }")
        lines.append(f"    {nid} ..> {fid} : calls x{count}")

    # Importers that never call it (re-exports)
    for rel in importers:
        if rel in callers or rel in definers:
            continue
        nid = node_id(rel)
        lines.append(f'    class {nid}["{mermaid_safe(short(rel))}"] {{')
        lines.append("        <<imports only>>")
        lines.append("    }")
        lines.append(f"    {nid} ..> {fid} : imports")

    header = (f"%% Function '{func_name}' in {repo_name}\n"
              f"%% defined in {len(definers)}, called in {len(callers)}, "
              f"imported by {len(importers)} file(s)\n")
    if len(ranked) > max_callers:
        header += f"%% showing top {max_callers} callers of {len(ranked)}\n"
    return header + "\n".join(lines) + "\n"


# ── 3. Class diagram (AST/regex scan — true UML: attrs, methods, inheritance) ─
#
# Unlike `deps` and `function`, this reads actual class bodies rather than
# just import/call sites. Python gets an exact AST parse; C# and JS/TS use
# brace-counting + regex (approximate — there's no parser for them here,
# same tradeoff as scan_function above). Base-class names are matched by
# NAME ONLY (source has no import resolution for base classes), so two
# same-named classes in different files can, rarely, be linked wrongly —
# acceptable for a best-effort diagram, not for correctness-critical use.

def _py_expr(node) -> str:
    """Best-effort string form of a Python expression/annotation."""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _scan_python_classes(text: str, rel: str) -> list:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_py_expr(b) for b in node.bases if _py_expr(b)]
        methods, attrs, seen = [], [], set()
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [a.arg for a in item.args.args if a.arg != "self"]
                ret = _py_expr(item.returns) if item.returns else ""
                vis = "-" if item.name.startswith("_") else "+"
                methods.append({"name": item.name, "params": params, "ret": ret, "vis": vis})
                if item.name == "__init__":
                    for sub in ast.walk(item):
                        target, ann = None, ""
                        if (isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Attribute)
                                and isinstance(sub.target.value, ast.Name) and sub.target.value.id == "self"):
                            target, ann = sub.target.attr, _py_expr(sub.annotation)
                        elif (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store)
                                and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
                            target = sub.attr
                        if target and target not in seen:
                            seen.add(target)
                            attrs.append({"name": target, "type": ann,
                                         "vis": "-" if target.startswith("_") else "+"})
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                nm = item.target.id
                if nm not in seen:
                    seen.add(nm)
                    attrs.append({"name": nm, "type": _py_expr(item.annotation),
                                 "vis": "-" if nm.startswith("_") else "+"})
        out.append({"name": node.name, "bases": bases, "attrs": attrs, "methods": methods,
                    "lang": "Python", "file": rel})
    return out


def _extract_braced_body(text: str, open_idx: int):
    """From the index of an opening '{', returns (body, close_idx) by brace counting."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i
    return text[open_idx + 1:], len(text)


CS_CLASS_HEAD = re.compile(
    r"(?:public|internal|private|protected|abstract|sealed|static|partial|\s)*"
    r"class\s+(\w+)\s*(?:<[^>]*>)?\s*(?::\s*([\w<>,.\s]+?))?\s*\{"
)
CS_METHOD = re.compile(
    r"(?:public|private|protected|internal|static|virtual|override|async|sealed|abstract|\s)+"
    r"[\w<>\[\],\.\?]+\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:\{|=>|;)"
)
CS_FIELD = re.compile(
    r"^\s*(?:public|private|protected|internal|static|readonly|const|\s)+"
    r"([\w<>\[\],\.\?]+)\s+(\w+)\s*(?:=[^;]*)?;", re.M
)


def _scan_csharp_classes(text: str, rel: str) -> list:
    text = _strip_comments(text, "C#")
    out = []
    for m in CS_CLASS_HEAD.finditer(text):
        name = m.group(1)
        bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()]
        body, _ = _extract_braced_body(text, m.end() - 1)

        methods = []
        for mm in CS_METHOD.finditer(body):
            mname = mm.group(1)
            if mname == name:      # constructor — same name as class, not a method
                continue
            params = [p.strip().split()[-1] for p in mm.group(2).split(",") if p.strip()]
            methods.append({"name": mname, "params": params, "ret": "", "vis": "+"})

        attrs, seen = [], set()
        for fm in CS_FIELD.finditer(body):
            ftype, fname = fm.group(1), fm.group(2)
            if fname in seen or ftype in ("class", "return", "using"):
                continue
            seen.add(fname)
            attrs.append({"name": fname, "type": ftype, "vis": "+"})

        out.append({"name": name, "bases": bases, "attrs": attrs[:12], "methods": methods[:20],
                    "lang": "C#", "file": rel})
    return out


JS_CLASS_HEAD = re.compile(r"class\s+(\w+)\s*(?:extends\s+([\w.]+))?\s*\{")
JS_METHOD = re.compile(r"(?:^|\n)\s*(?:static\s+|async\s+|get\s+|set\s+)*(\w+)\s*\(([^)]*)\)\s*\{")
JS_SKIP_METHOD_NAMES = {"if", "for", "while", "switch", "catch", "constructor"}


def _scan_js_classes(text: str, rel: str, lang: str) -> list:
    text = _strip_comments(text, lang)
    out = []
    for m in JS_CLASS_HEAD.finditer(text):
        name, base = m.group(1), m.group(2)
        body, _ = _extract_braced_body(text, m.end() - 1)

        methods = []
        for mm in JS_METHOD.finditer(body):
            mname = mm.group(1)
            if mname in JS_SKIP_METHOD_NAMES:
                continue
            params = [p.strip() for p in mm.group(2).split(",") if p.strip()]
            methods.append({"name": mname, "params": params, "ret": "", "vis": "+"})

        attrs, seen = [], set()
        ctor_m = re.search(r"constructor\s*\(([^)]*)\)\s*\{", body)
        if ctor_m:
            cbody, _ = _extract_braced_body(body, ctor_m.end() - 1)
            for fm in re.finditer(r"this\.(\w+)\s*=", cbody):
                if fm.group(1) not in seen:
                    seen.add(fm.group(1))
                    attrs.append({"name": fm.group(1), "type": "", "vis": "+"})
        # class-field syntax: `foo = 1;` at the top of the body (not inside a method)
        for fm in re.finditer(r"(?:^|\{)\s*(\w+)\s*=\s*[^=>][^;]*;", body):
            if fm.group(1) not in seen and fm.group(1) not in JS_SKIP_METHOD_NAMES:
                seen.add(fm.group(1))
                attrs.append({"name": fm.group(1), "type": "", "vis": "+"})

        out.append({"name": name, "bases": [base] if base else [],
                    "attrs": attrs[:12], "methods": methods[:20],
                    "lang": lang, "file": rel})
    return out


def _scan_file_classes(rel: str, full: Path) -> list:
    ext = Path(rel).suffix.lower()
    lang = SCANNABLE.get(ext)
    if lang not in ("Python", "C#", "JavaScript", "TypeScript"):
        return []       # Razor has no reliable class syntax to scan here
    try:
        text = full.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if "class" not in text:        # cheap prefilter
        return []
    if lang == "Python":
        return _scan_python_classes(text, rel)
    if lang == "C#":
        return _scan_csharp_classes(text, rel)
    return _scan_js_classes(text, rel, lang)


def scan_classes(repo_path: Path, repo_name: str, languages=None) -> list:
    """Every class found across the repo's scanned source files."""
    classes = []
    for rel, full in _iter_source_files(repo_path, repo_name, languages):
        classes.extend(_scan_file_classes(rel, full))
    return classes


def _class_index(classes: list) -> dict:
    """{name: class_dict}. Last definition wins on name collisions across files."""
    return {c["name"]: c for c in classes}


def _related_classes(classes: list, class_name: str):
    """target, direct ancestors, direct descendants, composed types (one hop)."""
    idx = _class_index(classes)
    if class_name not in idx:
        return None, [], [], []
    target = idx[class_name]
    ancestors = [idx[b] for b in target["bases"] if b in idx]
    descendants = [c for c in classes if class_name in c["bases"]]
    composed_names = {a["type"].strip("[]?") for a in target["attrs"] if a.get("type")}
    composed = [idx[n] for n in composed_names if n in idx and n != class_name]
    return target, ancestors, descendants, composed


def class_data(repo_name: str, repo_path: Path, class_name: str = None,
              rel_file: str = None, languages=None, top: int = 15) -> dict:
    """
    Assembles the data for one of three views, shared by the Mermaid and
    image emitters:
      - class_name given -> that class's hierarchy (ancestors/descendants/composed)
      - rel_file given    -> every class defined in that file
      - neither given     -> repo-wide, top N classes by member count
    """
    classes = scan_classes(repo_path, repo_name, languages)

    if class_name:
        target, ancestors, descendants, composed = _related_classes(classes, class_name)
        if target is None:
            return {"mode": "single", "found": False, "name": class_name}
        return {"mode": "single", "found": True, "target": target,
                "ancestors": ancestors, "descendants": descendants, "composed": composed}

    if rel_file:
        file_classes = [c for c in classes if c["file"] == rel_file]
        idx = _class_index(classes)
        in_file = {c["name"] for c in file_classes}
        externals = {}
        for c in file_classes:
            for b in c["bases"]:
                if b in idx and b not in in_file:
                    externals[b] = idx[b]
        return {"mode": "file", "file": rel_file, "classes": file_classes,
                "external_bases": list(externals.values())}

    ranked = sorted(classes, key=lambda c: -(len(c["methods"]) + len(c["attrs"])))[:top]
    idx = _class_index(classes)
    kept = {c["name"] for c in ranked}
    edges = [(c["name"], b) for c in ranked for b in c["bases"] if b in kept]
    return {"mode": "top", "classes": ranked, "edges": edges, "total": len(classes)}


def _class_node_id(c: dict) -> str:
    """Unique per file+class, so same-named classes in different files don't collide."""
    return node_id(f"{c.get('file', '')}::{c['name']}")


def _emit_uml_class(lines: list, c: dict, stereotype: str = None,
                    attr_limit: int = 8, method_limit: int = 10) -> str:
    """Writes one Mermaid `class { }` block in real UML member syntax. Returns its node id."""
    nid = _class_node_id(c)
    lines.append(f'    class {nid}["{mermaid_safe(c["name"])}"] {{')
    if stereotype:
        lines.append(f"        <<{stereotype}>>")
    for a in c["attrs"][:attr_limit]:
        t = f" {mermaid_safe(a['type'], 24)}" if a.get("type") else ""
        lines.append(f"        {a['vis']}{mermaid_safe(a['name'], 30)}{t}")
    for m in c["methods"][:method_limit]:
        params = ", ".join(mermaid_safe(p, 18) for p in m["params"][:5])
        ret = f" {mermaid_safe(m['ret'], 20)}" if m.get("ret") else ""
        lines.append(f"        {m['vis']}{mermaid_safe(m['name'], 30)}({params}){ret}")
    if len(c["attrs"]) > attr_limit or len(c["methods"]) > method_limit:
        lines.append("        …")
    lines.append("    }")
    return nid


def diagram_class(repo_name: str, repo_path: Path, class_name: str = None,
                  rel_file: str = None, languages=None, top: int = 15) -> str:
    d = class_data(repo_name, repo_path, class_name, rel_file, languages, top)

    if d["mode"] == "single":
        if not d["found"]:
            return (f"%% Class '{class_name}' not found in '{repo_name}'.\n"
                    f"classDiagram\n    class missing[\"{class_name} — not found\"]\n")
        lines = ["classDiagram", "    direction TB"]
        tid = _emit_uml_class(lines, d["target"], stereotype="target")
        for a in d["ancestors"]:
            aid = _emit_uml_class(lines, a)
            lines.append(f"    {aid} <|-- {tid}")
        for desc in d["descendants"]:
            did = _emit_uml_class(lines, desc)
            lines.append(f"    {tid} <|-- {did}")
        for comp in d["composed"]:
            cid = _emit_uml_class(lines, comp)
            lines.append(f"    {tid} --> {cid} : uses")
        header = (f"%% Class hierarchy for {class_name} ({repo_name})\n"
                  f"%% {len(d['ancestors'])} ancestor(s), {len(d['descendants'])} descendant(s), "
                  f"{len(d['composed'])} composed type(s)\n")
        return header + "\n".join(lines) + "\n"

    if d["mode"] == "file":
        if not d["classes"]:
            return (f"%% No classes found in {rel_file} ({repo_name}).\n"
                    f"classDiagram\n    class {node_id(rel_file)}[\"{short(rel_file)}\"]\n")
        lines = ["classDiagram", "    direction TB"]
        ids = {}
        for c in d["classes"]:
            ids[c["name"]] = _emit_uml_class(lines, c)
        for c in d["external_bases"]:
            ids[c["name"]] = _emit_uml_class(lines, c, stereotype="external")
        for c in d["classes"]:
            for b in c["bases"]:
                if b in ids:
                    lines.append(f"    {ids[b]} <|-- {ids[c['name']]}")
        header = f"%% Classes defined in {rel_file} ({repo_name})\n"
        return header + "\n".join(lines) + "\n"

    # mode == "top"
    if not d["classes"]:
        return f"%% No classes found in '{repo_name}'. Check --lang or --repo-path.\n"
    lines = ["classDiagram", "    direction TB"]
    ids = {}
    for c in d["classes"]:
        ids[c["name"]] = _emit_uml_class(lines, c, attr_limit=5, method_limit=6)
    for a, b in d["edges"]:
        lines.append(f"    {ids[b]} <|-- {ids[a]}")
    header = (f"%% Top {len(d['classes'])} of {d['total']} classes by member count — {repo_name}\n"
              f"%% Pass --class <Name> for one class's full hierarchy, "
              f"or --file <path> for one file's classes\n")
    return header + "\n".join(lines) + "\n"


def image_class(repo_name: str, repo_path: Path, out_path: Path, class_name: str = None,
                rel_file: str = None, languages=None, top: int = 15) -> Path:
    """Class hierarchy / file classes / top-N classes, as an image."""
    import networkx as nx

    d = class_data(repo_name, repo_path, class_name, rel_file, languages, top)

    def _empty(msg):
        fig, ax = _new_fig(9, 2.6)
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color=IMG_MUTED)
        ax.axis("off")
        return _save_image(fig, out_path)

    if d["mode"] == "single" and not d["found"]:
        return _empty(f"Class '{class_name}' not found")
    if d["mode"] == "file" and not d["classes"]:
        return _empty(f"No classes found in {short(rel_file, 2)}")
    if d["mode"] == "top" and not d["classes"]:
        return _empty("No classes found")

    G = nx.DiGraph()
    pos, labels, node_colors = {}, {}, {}

    def add(nid, label, x, y, color):
        G.add_node(nid); pos[nid] = (x, y); labels[nid] = label; node_colors[nid] = color

    if d["mode"] == "single":
        add("target", d["target"]["name"], 1.0, 0.0, IMG_ACCENT)
        for i, a in enumerate(d["ancestors"]):
            nid = f"anc{i}"; add(nid, a["name"], 1.0, 1.3 + i * 0.85, IMG_PRIMARY)
            G.add_edge(nid, "target")
        for i, dd in enumerate(d["descendants"]):
            nid = f"desc{i}"; add(nid, dd["name"], 1.0, -1.3 - i * 0.85, IMG_PRIMARY)
            G.add_edge("target", nid)
        for i, c in enumerate(d["composed"]):
            nid = f"comp{i}"
            add(nid, c["name"], 2.6, (i - (len(d["composed"]) - 1) / 2) * 0.85, IMG_MUTED)
            G.add_edge("target", nid)
        title = (f"Class {class_name}  ({repo_name})\n"
                 f"{len(d['ancestors'])} ancestor(s)  ·  {len(d['descendants'])} descendant(s)  ·  "
                 f"{len(d['composed'])} composed")
    elif d["mode"] == "file":
        classes = d["classes"] + d["external_bases"]
        ext_names = {c["name"] for c in d["external_bases"]}
        ids = {}
        for i, c in enumerate(classes):
            nid = f"c{i}"
            col = IMG_MUTED if c["name"] in ext_names else IMG_PRIMARY
            add(nid, c["name"], (i % 3) * 1.5, -(i // 3) * 1.0, col)
            ids[c["name"]] = nid
        for c in d["classes"]:
            for b in c["bases"]:
                if b in ids:
                    G.add_edge(ids[b], ids[c["name"]])
        title = f"Classes in {short(rel_file, 2)}  ({repo_name})"
    else:
        ids = {}
        cols = 4
        for i, c in enumerate(d["classes"]):
            nid = f"c{i}"
            add(nid, c["name"], (i % cols) * 1.7, -(i // cols) * 1.1, IMG_PRIMARY)
            ids[c["name"]] = nid
        for a, b in d["edges"]:
            if a in ids and b in ids:
                G.add_edge(ids[b], ids[a])
        title = f"Top {len(d['classes'])} of {d['total']} classes by size  ({repo_name})"

    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    w = max(9.0, (max(xs) - min(xs)) * 2.2 + 5)
    h = max(4.0, (max(ys) - min(ys)) + 3)

    fig, ax = _new_fig(w, min(h, 16))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=IMG_EDGE, width=1.2,
                           arrows=True, arrowsize=12, node_size=1300)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1000,
                           node_color=[node_colors[n] for n in G.nodes()], linewidths=0)
    for n in G.nodes():
        x, y = pos[n]
        ax.text(x, y, labels[n], ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
    ax.set_title(title, fontsize=11, pad=14)
    ax.axis("off")
    return _save_image(fig, out_path)


# ── 4. System diagram (whole-repo, module-level aggregation) ──────────────────
#
# At repo scale a file-level graph is unreadable (nopCommerce has ~3,600 C#
# files), so this aggregates to MODULE level — the first `module_depth` path
# segments — same idea as system_report.py's PDF views, but emitted here as
# a lightweight Mermaid/image diagram rather than a full report.

def _fetch_all_edges(repo_name: str) -> list:
    """All DEPENDS_ON file-pairs for a repo, straight from Kùzu."""
    try:
        from kb.graph.kuzu_store import get_connection, rows_to_dicts
        res = get_connection().execute("""
            MATCH (a:File {repository_name: $repo})-[:DEPENDS_ON]->(b:File)
            RETURN a.relative, b.relative
        """, {"repo": repo_name})
        return [(r[0], r[1]) for r in rows_to_dicts(res)]
    except Exception:
        return []


def system_data(repo_name: str, repo_path: Path, module_depth: int = 2,
                max_nodes: int = 30, languages=None) -> dict:
    edges = _fetch_all_edges(repo_name)
    classes = scan_classes(repo_path, repo_name, languages) if repo_path else []

    mod_edges = defaultdict(int)
    mod_files = defaultdict(set)
    for a, b in edges:
        ma, mb = module_of(a, module_depth), module_of(b, module_depth)
        mod_files[ma].add(a)
        mod_files[mb].add(b)
        if ma != mb:
            mod_edges[(ma, mb)] += 1

    mod_classes = defaultdict(int)
    for c in classes:
        mod_classes[module_of(c["file"], module_depth)] += 1

    touch = defaultdict(int)
    for (a, b), n in mod_edges.items():
        touch[a] += n
        touch[b] += n

    all_mods = set(touch) | set(mod_classes) | set(mod_files)
    ranked = sorted(all_mods, key=lambda m: -(touch[m] + mod_classes[m] + len(mod_files.get(m, ()))))
    kept = set(ranked[:max_nodes])

    return {
        "modules": [{"name": m, "files": len(mod_files.get(m, ())), "classes": mod_classes.get(m, 0)}
                    for m in ranked if m in kept],
        "edges": [(a, b, n) for (a, b), n in mod_edges.items() if a in kept and b in kept],
        "total_modules": len(all_mods),
        "total_files": len({f for pair in edges for f in pair}),
        "total_classes": len(classes),
    }


def diagram_system(repo_name: str, repo_path: Path, module_depth: int = 2,
                   max_nodes: int = 30, languages=None) -> str:
    d = system_data(repo_name, repo_path, module_depth, max_nodes, languages)
    if not d["modules"]:
        return (f"%% No data for '{repo_name}'. Check ingestion ran, "
                f"or pass --repo-path if classes should be included.\n")

    lines = ["classDiagram", "    direction TB"]
    for m in d["modules"]:
        nid = node_id(m["name"])
        lines.append(f'    class {nid}["{mermaid_safe(short(m["name"], 3))}"] {{')
        lines.append("        <<module>>")
        lines.append(f"        +{m['files']} files")
        if m["classes"]:
            lines.append(f"        +{m['classes']} classes")
        lines.append("    }")
    for a, b, n in d["edges"]:
        lines.append(f"    {node_id(a)} ..> {node_id(b)} : {n}")

    header = (f"%% System overview — {repo_name}\n"
              f"%% {len(d['modules'])} of {d['total_modules']} modules shown "
              f"(module depth {module_depth}), {d['total_files']} files, "
              f"{d['total_classes']} classes total\n")
    return header + "\n".join(lines) + "\n"


def image_system(repo_name: str, repo_path: Path, out_path: Path, module_depth: int = 2,
                 max_nodes: int = 30, languages=None) -> Path:
    import networkx as nx

    d = system_data(repo_name, repo_path, module_depth, max_nodes, languages)
    if not d["modules"]:
        fig, ax = _new_fig(9, 2.6)
        ax.text(0.5, 0.5, f"No data for '{repo_name}'", ha="center", va="center",
                fontsize=12, color=IMG_MUTED)
        ax.axis("off")
        return _save_image(fig, out_path)

    G = nx.DiGraph()
    file_counts = {}
    for m in d["modules"]:
        G.add_node(m["name"])
        file_counts[m["name"]] = m["files"]
    for a, b, n in d["edges"]:
        G.add_edge(a, b, weight=n)

    pos = nx.spring_layout(G, seed=42, k=1.4 / max(len(G.nodes()) ** 0.5, 1))
    sizes = [400 + 60 * file_counts.get(n, 0) for n in G.nodes()]

    fig, ax = _new_fig(13, 10)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=IMG_EDGE, width=0.8,
                           arrows=True, arrowsize=8, node_size=sizes, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=IMG_PRIMARY, linewidths=0, alpha=0.9)
    for n in G.nodes():
        x, y = pos[n]
        ax.text(x, y, short(n, 2), ha="center", va="center",
                fontsize=6.8, color="white", fontweight="bold")
    ax.set_title(f"System overview — {repo_name}\n"
                 f"{len(d['modules'])} of {d['total_modules']} modules  ·  "
                 f"{d['total_files']} files  ·  {d['total_classes']} classes",
                 fontsize=11, pad=14)
    ax.axis("off")
    return _save_image(fig, out_path)


# ── 5. Key point diagram (from ChromaDB) ──────────────────────────────────────

def keypoint_data(repo_name: str, kp_index: int, top_n: int = 10) -> dict:
    """Files most aligned to one key point — shared by both emitters."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll_name = collection_name_for(repo_name)
    available = [c.name for c in client.list_collections()]
    if coll_name not in available:
        sys.exit(f"Error: no collection '{coll_name}'. Available: {available}\n"
                 f"Run: python3 -m kb.vector.embeddings --repo {repo_name}")

    got = client.get_collection(coll_name).get(include=["metadatas", "embeddings"])
    kp_text, scored = None, []
    for meta, emb in zip(got["metadatas"], got["embeddings"]):
        if kp_index >= len(emb):
            sys.exit(f"Error: key point {kp_index} out of range "
                     f"(vectors have {len(emb)} dimensions / key points).")
        scored.append((float(emb[kp_index]), meta.get("relative"), meta))
        if meta.get("top_kp_index") == kp_index and not kp_text:
            kp_text = meta.get("top_kp")
    scored.sort(reverse=True)
    kp_text = _get_key_point_text(repo_name, kp_index) or kp_text or f"Key point {kp_index}"
    return {"kp_index": kp_index, "kp_text": kp_text,
            "top": scored[:top_n], "total": len(scored)}


def diagram_keypoint(repo_name: str, kp_index: int, top_n: int = 10) -> str:
    """
    UML diagram of the files most aligned to one repo key point.
    Vector dimension k = that file's similarity to key point k.
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    coll_name = collection_name_for(repo_name)
    available = [c.name for c in client.list_collections()]
    if coll_name not in available:
        sys.exit(f"Error: no collection '{coll_name}'. Available: {available}\n"
                 f"Run: python3 -m kb.vector.embeddings --repo {repo_name}")

    col = client.get_collection(coll_name)
    got = col.get(include=["metadatas", "embeddings"])

    kp_text = None
    scored = []
    for meta, emb in zip(got["metadatas"], got["embeddings"]):
        if kp_index >= len(emb):
            sys.exit(f"Error: key point {kp_index} out of range "
                     f"(vectors have {len(emb)} dimensions / key points).")
        scored.append((float(emb[kp_index]), meta.get("relative"), meta))
        if meta.get("top_kp_index") == kp_index and not kp_text:
            kp_text = meta.get("top_kp")

    if not scored:
        return f"%% No vectors in '{coll_name}'. Run the embed stage.\n"

    scored.sort(reverse=True)
    top = scored[:top_n]

    # Prefer the real key point text from PostgreSQL
    kp_text = _get_key_point_text(repo_name, kp_index) or kp_text or f"Key point {kp_index}"
    label = mermaid_safe(kp_text)

    lines = ["classDiagram", "    direction TB"]
    kid = f"kp_{kp_index}"
    lines.append(f'    class {kid}["Key Point {kp_index}"] {{')
    lines.append("        <<capability>>")
    lines.append(f"        {label}")
    lines.append("    }")

    for score, rel, meta in top:
        nid = node_id(rel)
        lines.append(f'    class {nid}["{mermaid_safe(short(rel))}"] {{')
        lines.append(f"        +{meta.get('rule_count', 0)} rules")
        lines.append(f"        similarity {score:.3f}")
        lines.append("    }")
        # UML realization: the file implements the capability
        lines.append(f"    {nid} ..|> {kid} : {score:.3f}")

    header = (f"%% Files most aligned to key point {kp_index} — {repo_name}\n"
              f"%% {comment_safe(kp_text)}\n"
              f"%% top {len(top)} of {len(scored)} files\n")
    return header + "\n".join(lines) + "\n"


def _get_key_point_text(repo_name: str, kp_index: int):
    try:
        from kb.relationaldb.initialize_db import get_connection as pg_conn
        conn = pg_conn()
    except Exception:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT point_text FROM key_points
                WHERE repository_name = %s AND point_index = %s
            """, (repo_name, kp_index))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None
    finally:
        conn.close()


# ── Output ────────────────────────────────────────────────────────────────────

# ── Image rendering (PNG / JPEG) ──────────────────────────────────────────────
#
# Mermaid is text; turning it into a raster needs Node's mermaid-cli, which
# pulls a headless Chromium. Rather than depend on that, these renderers draw
# the SAME relationships natively with matplotlib — no external tooling, and
# the data comes from the same *_data() functions the Mermaid emitters use.

IMG_PRIMARY = "#2e5c8a"
IMG_ACCENT  = "#c1663a"
IMG_MUTED   = "#8fa8bf"
IMG_EDGE    = "#c7cdd4"


def _new_fig(w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(w, h))


def _save_image(fig, out_path: Path, dpi: int = 150) -> Path:
    """Writes .png or .jpg/.jpeg based on the suffix."""
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        kw["pil_kwargs"] = {"quality": 92}
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", **kw)
    plt.close(fig)
    return out_path


def image_deps(repo_name, rel_file, out_path, depth=1, max_nodes=24) -> Path:
    """Dependency neighbourhood of one file, as an image."""
    import networkx as nx

    d = deps_data(repo_name, rel_file, depth)
    inc, out = d["direct_in"], d["direct_out"]

    if not inc and not out:
        fig, ax = _new_fig(9, 2.6)
        ax.text(0.5, 0.5, f"No DEPENDS_ON edges for {short(rel_file, 2)}",
                ha="center", va="center", fontsize=12, color=IMG_MUTED)
        ax.axis("off")
        return _save_image(fig, out_path)

    inc, out = inc[:max_nodes], out[:max_nodes]
    G = nx.DiGraph()
    target = short(rel_file, 2)
    G.add_node(target)
    pos = {target: (1.0, 0.0)}

    for i, p in enumerate(inc):
        n = f"in:{short(p, 2)}"
        G.add_node(n); G.add_edge(n, target)
        pos[n] = (0.0, (i - (len(inc) - 1) / 2) * 1.0)
    for i, p in enumerate(out):
        n = f"out:{short(p, 2)}"
        G.add_node(n); G.add_edge(target, n)
        pos[n] = (2.0, (i - (len(out) - 1) / 2) * 1.0)

    height = max(4.0, max(len(inc), len(out)) * 0.42 + 2.0)
    fig, ax = _new_fig(13, height)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=IMG_EDGE, width=1.2,
                           arrows=True, arrowsize=13, node_size=1500)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[target], node_size=2200,
                           node_color=IMG_ACCENT, node_shape="s", linewidths=0)
    others = [n for n in G.nodes() if n != target]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=others, node_size=900,
                           node_color=IMG_PRIMARY, linewidths=0)

    ax.text(*pos[target], target, ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")
    for n in others:
        x, y = pos[n]
        label = n.split(":", 1)[1]
        ha = "right" if x < 1 else "left"
        ax.text(x + (-0.06 if x < 1 else 0.06), y, label,
                ha=ha, va="center", fontsize=7.2, color="#33404d")

    ax.set_xlim(-1.15, 3.15)
    ax.set_title(f"Dependencies — {short(rel_file, 3)}  ({repo_name})\n"
                 f"left: {len(d['direct_in'])} dependents  ·  "
                 f"right: {len(d['direct_out'])} dependencies",
                 fontsize=11, pad=14)
    ax.axis("off")
    return _save_image(fig, out_path)


def image_function(repo_name, repo_path, func_name, out_path,
                   max_callers=18, languages=None) -> Path:
    """Where a function is defined and called, as an image."""
    import networkx as nx

    definers, importers, callers = scan_function(
        repo_path, repo_name, func_name, languages)

    if not definers and not callers:
        fig, ax = _new_fig(9, 2.6)
        ax.text(0.5, 0.5, f"'{func_name}' not found in any scanned source file",
                ha="center", va="center", fontsize=12, color=IMG_MUTED)
        ax.axis("off")
        return _save_image(fig, out_path)

    ranked = sorted(callers.items(), key=lambda x: -x[1])[:max_callers]
    G = nx.DiGraph()
    fn = f"{func_name}()"
    G.add_node(fn)
    pos = {fn: (1.0, 0.0)}

    defs = definers[:8]
    for i, p in enumerate(defs):
        n = f"def:{short(p, 2)}"
        G.add_node(n); G.add_edge(n, fn)
        pos[n] = (0.0, (i - (len(defs) - 1) / 2) * 1.0)
    for i, (p, cnt) in enumerate(ranked):
        n = f"call:{short(p, 2)}|{cnt}"
        G.add_node(n); G.add_edge(fn, n)
        pos[n] = (2.0, (i - (len(ranked) - 1) / 2) * 1.0)

    height = max(4.0, max(len(defs), len(ranked)) * 0.42 + 2.2)
    fig, ax = _new_fig(13, height)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=IMG_EDGE, width=1.1,
                           arrows=True, arrowsize=12, node_size=1400)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[fn], node_size=2600,
                           node_color=IMG_ACCENT, node_shape="s", linewidths=0)
    dn = [n for n in G.nodes() if n.startswith("def:")]
    cn = [n for n in G.nodes() if n.startswith("call:")]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=dn, node_size=950,
                           node_color=IMG_PRIMARY, node_shape="s", linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=cn, node_size=800,
                           node_color=IMG_MUTED, linewidths=0)

    ax.text(*pos[fn], fn, ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")
    for n in dn:
        x, y = pos[n]
        ax.text(x - 0.06, y, n.split(":", 1)[1] + "  (defines)",
                ha="right", va="center", fontsize=7.2, color="#33404d")
    for n in cn:
        x, y = pos[n]
        label, cnt = n.split(":", 1)[1].rsplit("|", 1)
        ax.text(x + 0.06, y, f"{label}  ×{cnt}",
                ha="left", va="center", fontsize=7.2, color="#33404d")

    ax.set_xlim(-1.25, 3.25)
    ax.set_title(f"Function {func_name}()  ({repo_name})\n"
                 f"defined in {len(definers)}  ·  called in {len(callers)} file(s)"
                 + (f"  ·  showing top {max_callers}" if len(callers) > max_callers else ""),
                 fontsize=11, pad=14)
    ax.axis("off")
    return _save_image(fig, out_path)


def image_keypoint(repo_name, kp_index, out_path, top_n=12) -> Path:
    """Files most aligned to a capability, as a ranked bar chart."""
    d = keypoint_data(repo_name, kp_index, top_n)
    top = d["top"]

    if not top:
        fig, ax = _new_fig(9, 2.6)
        ax.text(0.5, 0.5, "No vectors — run the embed stage",
                ha="center", va="center", fontsize=12, color=IMG_MUTED)
        ax.axis("off")
        return _save_image(fig, out_path)

    labels = [short(p, 2) for _, p, _ in top][::-1]
    scores = [s for s, _, _ in top][::-1]

    fig, ax = _new_fig(11, max(3.0, 0.42 * len(top) + 2.0))
    bars = ax.barh(range(len(top)), scores, color=IMG_PRIMARY)
    bars[-1].set_color(IMG_ACCENT)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_xlabel("Alignment to this capability", fontsize=9)
    ax.set_xlim(0, max(scores) * 1.14)
    for i, s in enumerate(scores):
        ax.text(s + max(scores) * 0.012, i, f"{s:.3f}", va="center", fontsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)

    kp = d["kp_text"]
    wrapped = kp if len(kp) < 84 else kp[:82] + "…"
    ax.set_title(f"KP{kp_index} — {wrapped}\n"
                 f"top {len(top)} of {d['total']} files  ({repo_name})",
                 fontsize=11, pad=14)
    return _save_image(fig, out_path)


def wrap_markdown(mermaid: str, title: str) -> str:
    return f"# {title}\n\n```mermaid\n{mermaid}```\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate UML (Mermaid) diagrams from the Groundwork knowledge base")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--type", required=True,
                        choices=["deps", "function", "class", "keypoint", "system"],
                        help="deps = file dependencies/dependents; "
                             "function = where a function is defined/called; "
                             "class = UML class diagram (attrs/methods/inheritance); "
                             "keypoint = files aligned to a key point; "
                             "system = whole-repo module-level overview")
    parser.add_argument("--file", default=None,
                        help="Target file (--type deps; also --type class for one file's classes)")
    parser.add_argument("--function", default=None, help="Function name (for --type function)")
    parser.add_argument("--class", dest="class_name", default=None,
                        help="Class name (--type class) — shows its full hierarchy")
    parser.add_argument("--kp", type=int, default=None, help="Key point index (for --type keypoint)")
    parser.add_argument("--repo-path", default=None, help="Repo location on disk")
    parser.add_argument("--depth", type=int, default=1,
                        help="Dependency hops to follow (--type deps, default 1)")
    parser.add_argument("--module-depth", type=int, default=2,
                        help="Path segments per module (--type system, default 2)")
    parser.add_argument("--max-nodes", type=int, default=30,
                        help="Max modules drawn (--type system, default 30)")
    parser.add_argument("--lang", default=None,
                        help="Restrict source scan to a language "
                             "(e.g. 'C#', 'Razor', 'JavaScript', 'Python') — "
                             "used by --type function/class/system")
    parser.add_argument("--top", type=int, default=10,
                        help="How many items to show (keypoint/function/class, default 10)")
    parser.add_argument("--format", default="mermaid",
                        choices=["mermaid", "png", "jpeg", "both"],
                        help="mermaid = Mermaid source (default); png/jpeg = image; "
                             "both = Mermaid file + image")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Image resolution (default: 150)")
    parser.add_argument("--out", default=None,
                        help="Output path. Extension is honoured for images; "
                             "defaults to diagrams/<repo>_<type>.<ext>")
    args = parser.parse_args()

    repo_path = resolve_repo_path(args.repo, args.repo_path) \
        if args.type in ("function", "class") else None

    if args.type == "deps":
        if not args.file:
            sys.exit("--type deps requires --file")
        mermaid = diagram_deps(args.repo, args.file, args.depth)
        title = f"Dependencies — {args.file}"

    elif args.type == "function":
        if not args.function:
            sys.exit("--type function requires --function")
        langs = [args.lang] if args.lang else None
        mermaid = diagram_function(args.repo, repo_path, args.function, args.top, langs)
        title = f"Function — {args.function}()"

    elif args.type == "class":
        if not args.class_name and not args.file:
            sys.exit("--type class requires --class <Name> or --file <path>")
        langs = [args.lang] if args.lang else None
        mermaid = diagram_class(args.repo, repo_path, args.class_name, args.file, langs, args.top)
        title = f"Class — {args.class_name}" if args.class_name else f"Classes — {args.file}"

    elif args.type == "system":
        repo_path = try_resolve_repo_path(args.repo, args.repo_path)
        langs = [args.lang] if args.lang else None
        mermaid = diagram_system(args.repo, repo_path, args.module_depth, args.max_nodes, langs)
        title = f"System overview — {args.repo}"

    else:
        if args.kp is None:
            sys.exit("--type keypoint requires --kp <index>")
        mermaid = diagram_keypoint(args.repo, args.kp, args.top)
        title = f"Key Point {args.kp} — {args.repo}"

    # ── Image output ──────────────────────────────────────────────────────
    want_img = args.format in ("png", "jpeg", "both")
    if want_img:
        ext = "png" if args.format in ("png", "both") else "jpeg"
        if args.out and Path(args.out).suffix.lower() in (".png", ".jpg", ".jpeg"):
            img_path = Path(args.out)
        else:
            stem = {"deps": Path(args.file or "file").stem,
                    "function": args.function or "function",
                    "class": args.class_name or (Path(args.file).stem if args.file else "classes"),
                    "keypoint": f"kp{args.kp}",
                    "system": "system"}[args.type]
            base = Path(args.out).with_suffix("") if args.out else \
                   Path("diagrams") / f"{args.repo}_{args.type}_{stem}"
            img_path = base.with_suffix(f".{ext}")

        if args.type == "deps":
            p = image_deps(args.repo, args.file, img_path, args.depth)
        elif args.type == "function":
            langs = [args.lang] if args.lang else None
            p = image_function(args.repo, repo_path, args.function, img_path,
                               max_callers=args.top, languages=langs)
        elif args.type == "class":
            langs = [args.lang] if args.lang else None
            p = image_class(args.repo, repo_path, img_path, args.class_name,
                            args.file, langs, args.top)
        elif args.type == "system":
            langs = [args.lang] if args.lang else None
            p = image_system(args.repo, repo_path, img_path, args.module_depth,
                             args.max_nodes, langs)
        else:
            p = image_keypoint(args.repo, args.kp, img_path, top_n=args.top)
        print(f"  ✓ Wrote {p}")

    # ── Mermaid output ────────────────────────────────────────────────────
    if args.format in ("mermaid", "both"):
        if args.out and not want_img:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(wrap_markdown(mermaid, title), encoding="utf-8")
            print(f"  ✓ Wrote {out}")
            print(f"    Renders in GitHub, Jupyter, or any Mermaid viewer.")
        elif args.format == "both":
            md = (Path(args.out).with_suffix(".md") if args.out
                  else Path("diagrams") / f"{args.repo}_{args.type}.md")
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(wrap_markdown(mermaid, title), encoding="utf-8")
            print(f"  ✓ Wrote {md}")
        else:
            print(mermaid)


if __name__ == "__main__":
    main()