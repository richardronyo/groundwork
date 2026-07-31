#!/usr/bin/env python3
"""
Groundwork — Diagrams (v2, rebuilt)

Two diagram types, on purpose — everything else from the previous version
(function/keypoint/system views) has been dropped for focus:

  deps   Dependency graph — real Mermaid `graph TD` (not classDiagram), from
         Kùzu DEPENDS_ON edges. Anchor on one file and follow N hops in both
         directions, or omit --file for a whole-repo view capped to the
         busiest files.

  class  UML class diagram for ONE FILE — every class it defines, with
         attributes (+/- visibility, type), methods (params, return type),
         and inheritance arrows. Base classes not defined in the same file
         show as unlabeled external stubs.

         Tries PostgreSQL first (kb.relationaldb.metadata.py now saves
         classes/attributes/methods there, not just counts) — if the file's
         been indexed, this needs no disk access and --repo-path is unused.
         Falls back to scanning the file directly from disk if nothing's
         indexed for it yet, in which case --repo-path matters. Python is
         exact (AST) either way; C# and JS/TS are regex + brace-counting
         (approximate — there's no real parser for them here), same on both
         paths since kb.relationaldb.parser.py and this file share that
         approach on purpose.

Run from the PROJECT ROOT.

Usage:
    python3 -m kb.diagram --repo flask --type deps --file flask/src/flask/app.py
    python3 -m kb.diagram --repo flask --type deps --file flask/src/flask/app.py --depth 2
    python3 -m kb.diagram --repo flask --type deps                    # whole-repo, busiest files
    python3 -m kb.diagram --repo flask --type deps --max-nodes 40
    python3 -m kb.diagram --repo flask --type class --file flask/src/flask/app.py
    python3 -m kb.diagram --repo flask --type class --file flask/src/flask/app.py --format png
    python3 -m kb.diagram --repo flask --type class --file flask/src/flask/app.py --repo-path ./repos/flask

Requirements:
    pip install kuzu networkx matplotlib
    pip install "psycopg[binary]" python-dotenv   # class type's DB-first lookup
"""

import argparse
import ast
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Shared helpers ──────────────────────────────────────────────────────────

LANGUAGE_BY_EXT = {
    ".py": "Python",
    ".cs": "C#",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
}


def node_id(text: str) -> str:
    """Short, stable, Mermaid-safe id for arbitrary text (paths, class names)."""
    return "n" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def mermaid_safe(text: str, max_len: int = 40) -> str:
    """Escapes quotes and clips length so a label can't break the diagram syntax."""
    text = (text or "").replace('"', "'").replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def short(path: str, keep: int = 2) -> str:
    """Last `keep` path segments, for compact labels — 'a/b/c/d.py' -> '…/c/d.py'."""
    parts = Path(path).parts
    if len(parts) <= keep:
        return path
    return ".../" + "/".join(parts[-keep:])


def wrap_markdown(mermaid: str, title: str) -> str:
    return f"# {title}\n\n```mermaid\n{mermaid}```\n"


def resolve_repo_path(repo_name: str, explicit: str = None) -> Path:
    """Finds the repo's directory on disk — needed for --type class (reads real source)."""
    if explicit:
        p = Path(explicit)
        if not p.is_dir():
            sys.exit(f"Error: --repo-path '{p}' is not a directory.")
        return p
    for cand in (Path("./repos") / repo_name, Path(repo_name)):
        if cand.is_dir():
            return cand
    sys.exit(f"Error: repo not found on disk. Pass --repo-path. Tried ./repos/{repo_name}, ./{repo_name}")


# ── Image helpers (shared look across both diagram types) ────────────────────

IMG_PRIMARY = "#2e5c8a"
IMG_ACCENT  = "#c1663a"
IMG_MUTED   = "#9aa5b1"
IMG_EDGE    = "#c7cdd4"


def _new_fig(w: float, h: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    return fig, ax


def _save_image(fig, out_path) -> Path:
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _empty_image(out_path, message: str) -> Path:
    fig, ax = _new_fig(9, 2.6)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color=IMG_MUTED)
    ax.axis("off")
    return _save_image(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DEPENDENCY GRAPH — graph-style Mermaid + image, from Kùzu DEPENDS_ON
# ═══════════════════════════════════════════════════════════════════════════

def fetch_edges(repo_name: str) -> list[tuple[str, str]]:
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


def neighborhood(edges: list[tuple[str, str]], file: str, depth: int):
    """
    BFS both directions from `file` up to `depth` hops to find which nodes
    belong in the picture, then keeps EVERY edge between those nodes (not
    just the tree edges BFS walked) — so cross-links between two files that
    are each within range still show up.
    """
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    visited = {file}
    frontier = {file}
    for _ in range(max(depth, 0)):
        next_frontier = set()
        for n in frontier:
            next_frontier |= adj.get(n, set())
        next_frontier -= visited
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier

    kept_edges = sorted({(a, b) for a, b in edges if a in visited and b in visited})
    return visited, kept_edges


def busiest_subgraph(edges: list[tuple[str, str]], max_nodes: int):
    """Whole-repo view: keeps the max_nodes files with the most DEPENDS_ON traffic."""
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    top = {n for n, _ in degree.most_common(max_nodes)}
    kept_edges = sorted({(a, b) for a, b in edges if a in top and b in top})
    return top, kept_edges


def deps_data(repo_name: str, file: str = None, depth: int = 1, max_nodes: int = 60) -> dict:
    all_edges = fetch_edges(repo_name)
    if not all_edges:
        return {"found": False, "nodes": set(), "edges": [], "total_files": 0}

    all_files = {a for a, b in all_edges} | {b for a, b in all_edges}

    if file:
        if file not in all_files:
            return {"found": False, "nodes": set(), "edges": [], "total_files": len(all_files)}
        nodes, edges = neighborhood(all_edges, file, depth)
    else:
        nodes, edges = busiest_subgraph(all_edges, max_nodes)

    return {"found": True, "nodes": nodes, "edges": edges,
            "total_files": len(all_files), "total_edges": len(all_edges)}


def diagram_deps(repo_name: str, file: str = None, depth: int = 1, max_nodes: int = 60) -> str:
    d = deps_data(repo_name, file, depth, max_nodes)
    if not d["nodes"] and not d["found"]:
        if file:
            return (f"%% '{file}' has no DEPENDS_ON edges in '{repo_name}' "
                    f"(not in the graph, or the deps stage hasn't run).\n")
        return f"%% No DEPENDS_ON edges found for '{repo_name}'. Has the deps stage run?\n"

    lines = ["graph TD"]
    ids = {n: node_id(n) for n in d["nodes"]}
    for n in d["nodes"]:
        label = mermaid_safe(short(n))
        # Stadium shape marks the file we anchored on; everyone else is a plain box.
        shape = f'{ids[n]}(["{label}"])' if n == file else f'{ids[n]}["{label}"]'
        lines.append(f"    {shape}")
    for a, b in d["edges"]:
        lines.append(f"    {ids[a]} --> {ids[b]}")

    if file:
        header = (f"%% Dependency graph — {file} ({repo_name})\n"
                  f"%% {depth} hop(s) — {len(d['nodes'])} files, {len(d['edges'])} edges\n")
    else:
        header = (f"%% Dependency graph — {repo_name}\n"
                  f"%% Busiest {len(d['nodes'])} of {d['total_files']} files by DEPENDS_ON traffic, "
                  f"{len(d['edges'])} of {d['total_edges']} edges shown\n")
    return header + "\n".join(lines) + "\n"


def image_deps(repo_name: str, out_path, file: str = None, depth: int = 1, max_nodes: int = 60) -> Path:
    import networkx as nx

    d = deps_data(repo_name, file, depth, max_nodes)
    if not d["nodes"] and not d["found"]:
        msg = (f"'{file}' has no DEPENDS_ON edges" if file
               else f"No DEPENDS_ON edges found for '{repo_name}'")
        return _empty_image(out_path, msg)

    G = nx.DiGraph()
    G.add_nodes_from(d["nodes"])
    G.add_edges_from(d["edges"])
    pos = nx.spring_layout(G, seed=42, k=1.3 / max(len(d["nodes"]) ** 0.5, 1))

    side = max(9.0, len(d["nodes"]) * 0.35)
    fig, ax = _new_fig(side, side * 0.72)
    colors = [IMG_ACCENT if n == file else IMG_PRIMARY for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=IMG_EDGE, width=1.0,
                           arrows=True, arrowsize=10, node_size=800)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=650, node_color=colors, linewidths=0)
    for n in G.nodes():
        x, y = pos[n]
        ax.text(x, y, short(n), ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold")

    title = (f"Dependency graph — {short(file)}  ({repo_name})\n"
             f"{depth} hop(s) · {len(d['nodes'])} files · {len(d['edges'])} edges") if file else \
            (f"Dependency graph — {repo_name}\n"
             f"busiest {len(d['nodes'])} of {d['total_files']} files · {len(d['edges'])} edges")
    ax.set_title(title, fontsize=11, pad=14)
    ax.axis("off")
    return _save_image(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════
# 2. UML CLASS DIAGRAM — one file's classes, attributes, methods, inheritance
# ═══════════════════════════════════════════════════════════════════════════
#
# Python is parsed exactly via `ast`. C# and JS/TS are regex + brace-counting
# — there's no real parser for them here, so treat those two as best-effort.

def _py_expr(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _scan_python_classes(text: str) -> list[dict]:
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
        out.append({"name": node.name, "bases": bases, "attrs": attrs, "methods": methods})
    return out


def _strip_comments(text: str, lang: str) -> str:
    if lang == "Python":
        return text
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def _extract_braced_body(text: str, open_idx: int):
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


def _scan_csharp_classes(text: str) -> list[dict]:
    text = _strip_comments(text, "C#")
    out = []
    for m in CS_CLASS_HEAD.finditer(text):
        name = m.group(1)
        bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()]
        body, _ = _extract_braced_body(text, m.end() - 1)

        methods = []
        for mm in CS_METHOD.finditer(body):
            mname = mm.group(1)
            if mname == name:
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

        out.append({"name": name, "bases": bases, "attrs": attrs[:12], "methods": methods[:20]})
    return out


JS_CLASS_HEAD = re.compile(r"class\s+(\w+)\s*(?:extends\s+([\w.]+))?\s*\{")
JS_METHOD = re.compile(r"(?:^|\n)\s*(?:static\s+|async\s+|get\s+|set\s+)*(\w+)\s*\(([^)]*)\)\s*\{")
JS_SKIP_METHOD_NAMES = {"if", "for", "while", "switch", "catch", "constructor"}


def _scan_js_classes(text: str, lang: str) -> list[dict]:
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
        for fm in re.finditer(r"(?:^|\{)\s*(\w+)\s*=\s*[^=>][^;]*;", body):
            if fm.group(1) not in seen and fm.group(1) not in JS_SKIP_METHOD_NAMES:
                seen.add(fm.group(1))
                attrs.append({"name": fm.group(1), "type": "", "vis": "+"})

        out.append({"name": name, "bases": [base] if base else [],
                    "attrs": attrs[:12], "methods": methods[:20]})
    return out


def scan_file_classes(path: Path) -> list[dict]:
    """Every class defined in ONE file: [{name, bases, attrs, methods}]."""
    lang = LANGUAGE_BY_EXT.get(path.suffix.lower())
    if lang not in ("Python", "C#", "JavaScript", "TypeScript"):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if "class" not in text:
        return []
    if lang == "Python":
        return _scan_python_classes(text)
    if lang == "C#":
        return _scan_csharp_classes(text)
    return _scan_js_classes(text, lang)


def fetch_db_classes(repo_name: str, file_label: str) -> list[dict] | None:
    """
    Classes/attributes/methods for one file, read from PostgreSQL — populated
    by kb.relationaldb.metadata.py, which now saves structure (not just
    counts) alongside the metrics. Returns None (not []) if Postgres is
    unreachable or has nothing indexed for this exact file, so the caller can
    tell "no data available" apart from "genuinely zero classes" and fall
    back to scanning the file from disk.
    """
    try:
        from kb.relationaldb.initialize_db import get_connection
        conn = get_connection()
    except Exception:
        return None

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.name, c.bases
                FROM classes c JOIN files f ON f.id = c.file_id
                WHERE f.repository_name = %s AND f.file_path = %s
                ORDER BY c.id
            """, (repo_name, file_label))
            class_rows = cur.fetchall()
            if not class_rows:
                return None  # not indexed (yet) — let the caller fall back to disk

            class_ids = [r[0] for r in class_rows]
            cur.execute(
                "SELECT class_id, name, type, visibility FROM class_attributes "
                "WHERE class_id = ANY(%s) ORDER BY id",
                (class_ids,),
            )
            attr_rows = cur.fetchall()

            cur.execute(
                "SELECT class_id, name, params, return_type, visibility FROM functions "
                "WHERE class_id = ANY(%s) ORDER BY id",
                (class_ids,),
            )
            method_rows = cur.fetchall()
    except Exception:
        return None
    finally:
        conn.close()

    classes = {cid: {"name": name, "bases": list(bases or []), "attrs": [], "methods": []}
              for cid, name, bases in class_rows}
    for class_id, name, type_, vis in attr_rows:
        classes[class_id]["attrs"].append({"name": name, "type": type_ or "", "vis": vis or "+"})
    for class_id, name, params, ret, vis in method_rows:
        classes[class_id]["methods"].append({"name": name, "params": list(params or []),
                                            "ret": ret or "", "vis": vis or "+"})
    return list(classes.values())


def get_classes(repo_name: str, file_label: str, repo_path: Path = None):
    """
    Classes for one file — tries PostgreSQL first (fast, and doesn't require
    the repo to even be on disk), falls back to scanning the file directly
    if nothing's indexed for it yet.

    Returns (classes, source, file_path):
      source    "db" or "disk"
      file_path only set when disk was actually touched (used for "file not
                found" messaging) — None when PostgreSQL answered it, since
                then there was no need to resolve a path on disk at all.
    """
    db_classes = fetch_db_classes(repo_name, file_label)
    if db_classes is not None:
        return db_classes, "db", None

    if repo_path is None:
        repo_path = resolve_repo_path(repo_name)   # may sys.exit if not found anywhere
    file_path = repo_path.parent / file_label
    return scan_file_classes(file_path), "disk", file_path


def _emit_uml_class(lines: list, c: dict, stereotype: str = None,
                    attr_limit: int = 10, method_limit: int = 14) -> str:
    nid = node_id(c["name"])
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


def diagram_class(classes: list[dict], file_label: str, source: str = "disk",
                  file_path: Path = None) -> str:
    if not classes:
        missing = file_path is not None and not file_path.exists()
        return (f"%% No classes found in {file_label}{' (file not found)' if missing else ''}.\n"
                f"classDiagram\n    class {node_id(file_label)}[\"{short(file_label)}\"]\n")

    by_name = {c["name"]: c for c in classes}
    lines = ["classDiagram", "    direction TB"]
    ids = {}
    for c in classes:
        ids[c["name"]] = _emit_uml_class(lines, c)

    externals = {}
    for c in classes:
        for b in c["bases"]:
            if b not in by_name and b not in externals:
                externals[b] = {"name": b, "attrs": [], "methods": []}
    for name, stub in externals.items():
        ids[name] = _emit_uml_class(lines, stub, stereotype="external")

    for c in classes:
        for b in c["bases"]:
            if b in ids:
                lines.append(f"    {ids[b]} <|-- {ids[c['name']]}")

    header = (f"%% Classes in {file_label}  (source: {source})\n"
              f"%% {len(classes)} class(es) defined, {len(externals)} external base(s)\n")
    return header + "\n".join(lines) + "\n"


def _uml_box_lines(c: dict, attr_limit: int = 10, method_limit: int = 14):
    """Header/attribute/method text lines for one class box — same truncation
    limits as _emit_uml_class so the image and the Mermaid version agree."""
    attrs = c.get("attrs", [])[:attr_limit]
    attr_lines = []
    for a in attrs:
        type_suffix = f": {a['type']}" if a.get("type") else ""
        attr_lines.append(f"{a['vis']}{a['name']}{type_suffix}")
    if len(c.get("attrs", [])) > attr_limit:
        attr_lines.append("…")

    methods = c.get("methods", [])[:method_limit]
    method_lines = []
    for m in methods:
        params = ", ".join(m.get("params", [])[:5])
        ret = f": {m['ret']}" if m.get("ret") else ""
        method_lines.append(f"{m['vis']}{m['name']}({params}){ret}")
    if len(c.get("methods", [])) > method_limit:
        method_lines.append("…")

    return c["name"], attr_lines, method_lines


def image_class(classes: list[dict], file_label: str, out_path, source: str = "disk",
                file_path: Path = None) -> Path:
    """
    Real UML class boxes — header / attributes / methods compartments, same
    shape as any standard UML tool — not a node-link graph. Classes stack in
    layers by inheritance depth (base classes at top), with a hollow-triangle
    generalization arrow from each subclass up to its parent.
    """
    from matplotlib.patches import Rectangle, FancyArrowPatch

    if not classes:
        missing = file_path is not None and not file_path.exists()
        msg = f"File not found: {file_label}" if missing else f"No classes found in {short(file_label)}"
        return _empty_image(out_path, msg)

    by_name = {c["name"]: c for c in classes}
    externals = sorted({b for c in classes for b in c["bases"] if b not in by_name})
    boxes = [{**c, "external": False} for c in classes] + \
            [{"name": b, "attrs": [], "methods": [], "bases": [], "external": True} for b in externals]
    box_by_name = {b["name"]: b for b in boxes}

    # ── Size each box from its actual text content ──────────────────────────
    FONT_SIZE, CHAR_W, LINE_H, HEADER_H = 9, 0.10, 0.22, 0.34
    PAD_X, MIN_W = 0.16, 1.6

    for b in boxes:
        header, attr_lines, method_lines = _uml_box_lines(b)
        b["_header"], b["_attrs"], b["_methods"] = header, attr_lines, method_lines
        widest = max([len(header)] + [len(l) for l in attr_lines + method_lines] or [len(header)])
        b["_w"] = max(MIN_W, widest * CHAR_W + PAD_X * 2)
        if b["external"]:
            b["_attr_h"] = b["_method_h"] = 0.0
        else:
            # Every real class shows both compartments even when empty, same
            # convention as the reference diagram — a thin empty section
            # rather than no section at all.
            b["_attr_h"] = max(len(attr_lines), 1) * LINE_H if attr_lines else LINE_H * 0.7
            b["_method_h"] = max(len(method_lines), 1) * LINE_H if method_lines else LINE_H * 0.7
        b["_h"] = HEADER_H + b["_attr_h"] + b["_method_h"]

    # ── Layer by inheritance depth: base classes at layer 0, each subclass
    # one layer below the deepest base it extends — bases end up at the top.
    def layer_of(name: str, trail: frozenset = frozenset()) -> int:
        if name in trail:
            return 0   # inheritance cycle guard — shouldn't happen, but don't hang
        b = box_by_name.get(name)
        bases_in_set = [x for x in (b.get("bases") or []) if x in box_by_name] if b else []
        if not b or b["external"] or not bases_in_set:
            return 0
        return 1 + max(layer_of(x, trail | {name}) for x in bases_in_set)

    layers: dict[int, list[dict]] = defaultdict(list)
    for b in boxes:
        layers[layer_of(b["name"])].append(b)

    GAP_X, GAP_Y = 0.55, 1.0
    positions: dict[str, tuple[float, float]] = {}
    y_cursor, max_row_w = 0.0, 0.0
    for depth in sorted(layers):
        row = layers[depth]
        row_w = sum(b["_w"] for b in row) + GAP_X * (len(row) - 1)
        max_row_w = max(max_row_w, row_w)
        row_h = max(b["_h"] for b in row)
        x = -row_w / 2
        for b in row:
            positions[b["name"]] = (x + b["_w"] / 2, -(y_cursor + row_h / 2))
            x += b["_w"] + GAP_X
        y_cursor += row_h + GAP_Y

    # ── Draw ──────────────────────────────────────────────────────────────
    HEADER_COLOR, EXTERNAL_COLOR, BORDER = "#f2cf5b", "#e3e3e3", "#20202a"
    fig, ax = _new_fig(max(9.0, max_row_w + 2), max(5.0, y_cursor + 2))

    for b in boxes:
        cx, cy = positions[b["name"]]
        w, h = b["_w"], b["_h"]
        x0, y0 = cx - w / 2, cy - h / 2
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                               edgecolor=BORDER, linewidth=1.1, zorder=2))
        header_top = y0 + h
        ax.add_patch(Rectangle((x0, header_top - HEADER_H), w, HEADER_H,
                               facecolor=EXTERNAL_COLOR if b["external"] else HEADER_COLOR,
                               edgecolor=BORDER, linewidth=1.1, zorder=3))
        ax.text(cx, header_top - HEADER_H / 2, b["_header"], ha="center", va="center",
               fontsize=FONT_SIZE, fontweight="bold", family="monospace", zorder=4)

        if not b["external"]:
            attr_top = header_top - HEADER_H
            ax.plot([x0, x0 + w], [attr_top, attr_top], color=BORDER, linewidth=1.0, zorder=3)
            ty = attr_top - LINE_H * 0.6
            for line in b["_attrs"]:
                ax.text(x0 + PAD_X, ty, line, ha="left", va="center",
                       fontsize=FONT_SIZE - 1, family="monospace", zorder=4)
                ty -= LINE_H

            method_top = attr_top - b["_attr_h"]
            ax.plot([x0, x0 + w], [method_top, method_top], color=BORDER, linewidth=1.0, zorder=3)
            ty = method_top - LINE_H * 0.6
            for line in b["_methods"]:
                ax.text(x0 + PAD_X, ty, line, ha="left", va="center",
                       fontsize=FONT_SIZE - 1, family="monospace", zorder=4)
                ty -= LINE_H

    # Generalization arrows: child -> parent, hollow triangle at the parent end.
    for c in classes:
        for base in c["bases"]:
            if base not in positions:
                continue
            cx, cy = positions[c["name"]]
            px, py = positions[base]
            child_h, parent_h = box_by_name[c["name"]]["_h"], box_by_name[base]["_h"]
            start = (cx, cy + child_h / 2)
            end = (px, py - parent_h / 2)
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=16,
                facecolor="white", edgecolor=BORDER, linewidth=1.1, zorder=1,
            ))

    ax.set_title(f"Classes in {short(file_label)}  ({source})\n"
                 f"{len(classes)} class(es) · {len(externals)} external base(s)",
                 fontsize=11, pad=14)
    ax.set_xlim(-max_row_w / 2 - 1, max_row_w / 2 + 1)
    ax.set_ylim(-y_cursor - 1, 1)
    ax.axis("off")
    return _save_image(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Groundwork — dependency graphs and UML class diagrams")
    ap.add_argument("--repo", required=True, help="Repository name")
    ap.add_argument("--type", required=True, choices=["deps", "class"],
                    help="deps = dependency graph (Kùzu); class = UML class diagram for one file")
    ap.add_argument("--file", default=None,
                    help="For deps: anchor file (omit for whole-repo view). "
                         "For class: REQUIRED — the file to diagram.")
    ap.add_argument("--depth", type=int, default=1, help="deps: hops to follow from --file (default: 1)")
    ap.add_argument("--max-nodes", type=int, default=60,
                    help="deps: files shown in whole-repo view when --file is omitted (default: 60)")
    ap.add_argument("--repo-path", default=None,
                    help="class: repo location on disk — only needed as a fallback for "
                         "files not yet indexed in PostgreSQL (see kb.relationaldb.metadata)")
    ap.add_argument("--format", default="mermaid", choices=["mermaid", "png", "jpeg", "both"])
    ap.add_argument("--out", default=None, help="Output path (.md or .png/.jpg)")
    args = ap.parse_args()

    if args.type == "class" and not args.file:
        sys.exit("--type class requires --file <path>")

    if args.type == "deps":
        mermaid = diagram_deps(args.repo, args.file, args.depth, args.max_nodes)
        title = f"Dependencies — {args.file}" if args.file else f"Dependencies — {args.repo}"
        file_path, file_label = None, args.file
    else:
        file_label = args.file
        explicit_repo_path = None
        if args.repo_path:
            explicit_repo_path = Path(args.repo_path)
            if not explicit_repo_path.is_dir():
                sys.exit(f"Error: --repo-path '{explicit_repo_path}' is not a directory.")
        # Tries PostgreSQL first (kb.relationaldb.metadata.py populates it) — if
        # that file's been indexed, this needs no disk access at all, so
        # --repo-path only matters as a fallback for files that aren't in the DB yet.
        classes, source, file_path = get_classes(args.repo, file_label, explicit_repo_path)
        mermaid = diagram_class(classes, file_label, source, file_path)
        title = f"Classes — {file_label}"

    want_img = args.format in ("png", "jpeg", "both")
    if want_img:
        ext = "png" if args.format in ("png", "both") else "jpeg"
        if args.out and Path(args.out).suffix.lower() in (".png", ".jpg", ".jpeg"):
            img_path = Path(args.out)
        else:
            stem = Path(args.file).stem if args.file else args.repo
            base = Path(args.out).with_suffix("") if args.out else \
                   Path("diagrams") / f"{args.repo}_{args.type}_{stem}"
            img_path = base.with_suffix(f".{ext}")

        if args.type == "deps":
            p = image_deps(args.repo, img_path, args.file, args.depth, args.max_nodes)
        else:
            p = image_class(classes, file_label, img_path, source, file_path)
        print(f"  ✓ Wrote {p}")

    if args.format in ("mermaid", "both"):
        if args.out and not want_img:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(wrap_markdown(mermaid, title), encoding="utf-8")
            print(f"  ✓ Wrote {out}")
            print("    Renders in GitHub, Jupyter, or any Mermaid viewer.")
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