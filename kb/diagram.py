#!/usr/bin/env python3
"""
Groundwork — UML Diagram Generator

Renders relationships from the knowledge base as Mermaid UML class diagrams
(which GitHub, Jupyter, and most Markdown viewers render natively).

Three diagram types:

  deps      Which files depend on a file, and which it depends on.
            Source: Kùzu DEPENDS_ON edges (both directions).

  function  Which files define and call a given function.
            Source: AST scan of the repo source. The graph is FILE-level —
            it has no Function nodes or CALLS edges — so this reads the
            source that the knowledge base indexes.

  keypoint  Which files align most closely to a repo key point.
            Source: ChromaDB vectors (dimension k = similarity to key point k).

Run from the PROJECT ROOT.

Usage:
    python3 -m kb.diagrams --repo flask --type deps --file src/flask/app.py
    python3 -m kb.diagrams --repo flask --type function --function make_response
    python3 -m kb.diagrams --repo flask --type keypoint --kp 3
    python3 -m kb.diagrams --repo flask --type keypoint --kp 3 --out diagrams/kp3.md

Requirements:
    pip install kuzu chromadb psycopg python-dotenv
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
        for ext in SCANNABLE:
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


# ── 3. Key point diagram (from ChromaDB) ──────────────────────────────────────

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
                        choices=["deps", "function", "keypoint"],
                        help="deps = file dependencies/dependents; "
                             "function = where a function is defined/called; "
                             "keypoint = files aligned to a key point")
    parser.add_argument("--file", default=None, help="Target file (for --type deps)")
    parser.add_argument("--function", default=None, help="Function name (for --type function)")
    parser.add_argument("--kp", type=int, default=None, help="Key point index (for --type keypoint)")
    parser.add_argument("--repo-path", default=None, help="Repo location on disk")
    parser.add_argument("--depth", type=int, default=1,
                        help="Dependency hops to follow (--type deps, default 1)")
    parser.add_argument("--lang", default=None,
                        help="Restrict --type function scan to a language "
                             "(e.g. 'C#', 'Razor', 'JavaScript', 'Python')")
    parser.add_argument("--top", type=int, default=10,
                        help="How many files to show (keypoint/function, default 10)")
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
        if args.type == "function" else None

    if args.type == "deps":
        if not args.file:
            sys.exit("--type deps requires --file")
        mermaid = diagram_deps(args.repo, args.file, args.depth)
        title = f"Dependencies — {args.file}"

    elif args.type == "function":
        if not args.function:
            sys.exit("--type function requires --function")
        repo_path = resolve_repo_path(args.repo, args.repo_path)
        langs = [args.lang] if args.lang else None
        mermaid = diagram_function(args.repo, repo_path, args.function, args.top, langs)
        title = f"Function — {args.function}()"

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
                    "keypoint": f"kp{args.kp}"}[args.type]
            base = Path(args.out).with_suffix("") if args.out else \
                   Path("diagrams") / f"{args.repo}_{args.type}_{stem}"
            img_path = base.with_suffix(f".{ext}")

        if args.type == "deps":
            p = image_deps(args.repo, args.file, img_path, args.depth)
        elif args.type == "function":
            langs = [args.lang] if args.lang else None
            p = image_function(args.repo, repo_path, args.function, img_path,
                               max_callers=args.top, languages=langs)
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