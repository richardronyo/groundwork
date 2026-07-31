#!/usr/bin/env python3
"""
Groundwork — System Report

Generates repository-wide architecture diagrams and writes them to a PDF with
descriptive text derived from the actual data (not boilerplate).

Four views:

  1. Module dependency graph   Which modules import which, aggregated from
                               file-level DEPENDS_ON edges in Kùzu.
  2. Function relationships    Functions defined in one file and called in
                               others, from an AST/regex scan of the source.
  3. Key point landscape       Which files cluster around which capability,
                               from the ChromaDB vectors.
  4. Module responsibility     Which module owns which functions, from the
                               PostgreSQL metrics plus the function scan.

At repository scale a file-level graph is unreadable — nopCommerce has ~3,600
C# files. Every view therefore aggregates to MODULE level (a configurable path
depth) and caps how many nodes are drawn, keeping the busiest ones.

The four main figures are unchanged. The drill-down APPENDIX lost its
per-capability (KP) detail image — diagram.py no longer generates keypoint
diagrams — and its per-function detail was swapped for a class detail (shows
the classes defined in the file that owns the most widely-shared function),
since diagram.py no longer generates function diagrams either.

Run from the PROJECT ROOT.

Usage:
    python3 -m system_report --repo nopCommerce
    python3 -m system_report --repo nopCommerce --out reports/nop.pdf
    python3 -m system_report --repo flask --module-depth 2 --max-nodes 25

Requirements:
    pip install matplotlib networkx reportlab psycopg kuzu chromadb python-dotenv
"""

import os
import re
import sys
import json
import argparse
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import networkx as nx

from dotenv import load_dotenv

# Build on kb/diagram.py rather than duplicating it, where it still overlaps.
# NOTE: diagram.py was rebuilt to cover only two diagram types (deps, class);
# it no longer has function/keypoint diagram generators, a ChromaDB collection
# helper, or the old scan_function/_scan_regex/_iter_source_files primitives.
# Everything below that used to come from there is now either reimplemented
# locally (collection_name_for) or was a dead import to begin with — the four
# main report figures were already self-contained; only the appendix's
# per-function and per-capability drill-down images depended on kb.diagram,
# and those are adjusted below (see build_appendix_images).
from kb.diagram import (
    LANGUAGE_BY_EXT as SCANNABLE,
    _strip_comments,
    short,
    image_deps,
    image_class,
)

load_dotenv()

CHROMA_DB_PATH    = "./chroma_db"
CHROMA_COLLECTION = "groundwork"

# Consistent palette across every figure
C_PRIMARY = "#2e5c8a"
C_ACCENT  = "#c1663a"
C_MUTED   = "#9aa5b1"
C_EDGE    = "#c7cdd4"


def collection_name_for(repo_name: str) -> str:
    """ChromaDB collection naming convention — used to live in kb.diagram;
    reimplemented here since diagram.py no longer touches ChromaDB at all."""
    safe = "".join(c if c.isalnum() else "_" for c in (repo_name or "").lower())
    return f"groundwork_{safe}"


# ── Data access ───────────────────────────────────────────────────────────────

def module_of(path: str, depth: int = 2) -> str:
    """
    The module a file belongs to: its first `depth` path segments AFTER the
    repo name.

    Every relative path in the knowledge base is now "<repo>/...", so
    path.parts[0] is always the repo name, not part of the module structure —
    without stripping it, depth=2 would put every file under "<repo>/src" in
    the SAME module regardless of subdirectory, collapsing cross-module
    figures to nothing.

    'nopCommerce/src/Libraries/Nop.Services/Catalog/ProductService.cs' (depth=2)
      -> 'src/Libraries'
    """
    parts = Path(path).parts
    if len(parts) <= 1:
        return "(root)"
    body = parts[1:]                       # drop the repo-name segment
    if len(body) <= 1:
        return "(root)"
    return "/".join(body[:min(depth, len(body) - 1)])


def fetch_edges(repo: str):
    """All DEPENDS_ON edges as (source, target) file pairs."""
    try:
        from kb.graph.kuzu_store import get_connection, rows_to_dicts
        res = get_connection().execute("""
            MATCH (a:File {repository_name: $repo})-[:DEPENDS_ON]->(b:File)
            RETURN a.relative, b.relative
        """, {"repo": repo})
        return [(r[0], r[1]) for r in rows_to_dicts(res)]
    except Exception as e:
        print(f"  (graph unavailable: {e})")
        return []


def fetch_files(repo: str):
    """File metrics from PostgreSQL: [{path, language, functions, classes, lines}]."""
    try:
        from kb.relationaldb.initialize_db import get_connection
        conn = get_connection()
    except Exception as e:
        print(f"  (PostgreSQL unavailable: {e})")
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT file_path, language, classes, functions, methods, lines
                FROM files WHERE repository_name = %s
            """, (repo,))
            return [{"path": r[0], "language": r[1], "classes": r[2] or 0,
                     "functions": r[3] or 0, "methods": r[4] or 0, "lines": r[5] or 0}
                    for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def fetch_key_points(repo: str):
    try:
        from kb.relationaldb.initialize_db import get_connection, load_key_points_from_db
        conn = get_connection()
        try:
            return load_key_points_from_db(conn, repo)
        finally:
            conn.close()
    except Exception:
        return []


def fetch_vectors(repo: str):
    """{file_path: [scores]} from the repo's ChromaDB collection."""
    try:
        import chromadb
        name = collection_name_for(repo)      # from kb.diagram
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        if name not in [c.name for c in client.list_collections()]:
            return {}
        got = client.get_collection(name).get(include=["embeddings"])
        return {i: [float(x) for x in e]
                for i, e in zip(got["ids"], got["embeddings"])}
    except Exception as e:
        print(f"  (vectors unavailable: {e})")
        return {}


def fetch_repo_metadata(repo: str, repo_path: Path) -> dict:
    """Gather repository metadata for the summary."""
    metadata = {
        "name": repo,
        "primary_language": None,
        "file_count": 0,
        "line_count": 0,
        "description": None,
        "dependencies": [],
        "key_points": [],
        "main_modules": [],
    }
    
    # Get file stats
    files = fetch_files(repo)
    if files:
        metadata["file_count"] = len(files)
        metadata["line_count"] = sum(f.get("lines", 0) for f in files)
        
        # Find primary language
        lang_count = Counter(f.get("language") or "Other" for f in files)
        if lang_count:
            metadata["primary_language"] = lang_count.most_common(1)[0][0]
    
    # Try to get repository description from key points or README
    key_points = fetch_key_points(repo)
    if key_points:
        metadata["key_points"] = key_points[:5]  # First 5 key points
        # Use first key point as a rough description
        if key_points:
            metadata["description"] = key_points[0]
    
    # Get module structure
    if files:
        mods = Counter(module_of(f["path"], 2) for f in files)
        metadata["main_modules"] = [m for m, _ in mods.most_common(8)]
    
    # Try to read README
    if repo_path and repo_path.is_dir():
        for readme in repo_path.glob("README*"):
            try:
                content = readme.read_text(encoding="utf-8", errors="ignore")
                # Try to extract first paragraph
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                for line in lines:
                    if len(line) > 40 and not line.startswith("#"):
                        metadata["description"] = line[:200]
                        break
                break
            except Exception:
                pass
    
    return metadata


# ── Function scanning (repo-wide inventory; independent of kb.diagram) ────────

# Definition patterns per language. The CALL-site scanning and comment/string
# stripping reuse kb.diagram's SCANNABLE-equivalent (LANGUAGE_BY_EXT) and
# _strip_comments — only "find every definition" is new here, since a system
# report needs the whole inventory rather than one named function.
DEF_PATTERNS = {
    ".py":  [r"^\s*(?:async\s+)?def\s+(\w+)\s*\("],
    ".cs":  [r"(?:public|private|protected|internal)\s+"
             r"(?:static\s+|virtual\s+|override\s+|async\s+|sealed\s+|partial\s+)*"
             r"[\w<>\[\],\.\?]+\s+(\w+)\s*(?:<[^>]*>)?\s*\("],
    ".js":  [r"function\s+(\w+)\s*\(", r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("],
    ".ts":  [r"function\s+(\w+)\s*\(", r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("],
}
SKIP_NAMES = {"if", "for", "while", "switch", "catch", "using", "lock", "return",
              "get", "set", "new", "foreach", "do", "try", "else"}


def scan_repo_functions(repo_path: Path, repo: str, files, max_files: int = 600):
    """
    Repo-wide function inventory: {name: {"defs": [paths], "calls": {path: n}}}.

    Uses kb.diagram's SCANNABLE-equivalent and _strip_comments so the language
    rules stay in one place. Scans the largest files first and caps the total —
    a full pass over 3,600 C# files is slow and the long tail adds little to a
    system-level view.
    """
    ranked = sorted(files, key=lambda f: -f.get("lines", 0))[:max_files]
    on_disk = []
    index = defaultdict(lambda: {"defs": [], "calls": defaultdict(int)})

    # Pass 1 — every definition
    for meta in ranked:
        rel = meta["path"]
        ext = Path(rel).suffix.lower()
        if ext not in SCANNABLE or ext not in DEF_PATTERNS:
            continue
        # `rel` starts with the repo name (e.g. "flask/src/flask/app.py");
        # repo_path already ends in the repo name, so join against its PARENT
        # to avoid doubling it — same convention as diagram.py and
        # file_dependencies.py/extract_business_rules.py.
        full = repo_path.parent / rel
        if not full.is_file():
            continue
        try:
            raw = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = _strip_comments(raw, SCANNABLE[ext])   # from kb.diagram
        on_disk.append((rel, ext, text))
        for pat in DEF_PATTERNS[ext]:
            for m in re.finditer(pat, text, re.M):
                name = m.group(1)
                if name and name not in SKIP_NAMES and not name.startswith("_"):
                    if rel not in index[name]["defs"]:
                        index[name]["defs"].append(rel)

    # Pass 2 — call sites for the names we saw defined
    known = {n for n, v in index.items() if v["defs"]}
    for rel, ext, text in on_disk:
        for name in known:
            if name not in text:
                continue
            n = len(re.findall(rf"(?<![\w.]){re.escape(name)}\s*\(", text)) + \
                len(re.findall(rf"\.{re.escape(name)}\s*\(", text))
            if rel in index[name]["defs"]:
                n = max(n - 1, 0)      # don't count the declaration itself
            if n:
                index[name]["calls"][rel] += n

    return {n: {"defs": v["defs"], "calls": dict(v["calls"])}
            for n, v in index.items() if v["defs"]}


# ── Description writers (text derived from the data) ──────────────────────────

def describe_module_deps(f):
    out = []
    if not f.get("module_edges"):
        out.append("No cross-module dependencies were recorded. Either the "
                   "<i>deps</i> stage has not run, or this repository's import style "
                   "does not resolve to file paths — C# <font face='Courier'>using</font> "
                   "statements name namespaces rather than files, so they often "
                   "resolve to nothing.")
        return out
    out.append(
        f"The knowledge base holds <b>{f['file_edges']} file-level import edges</b>, "
        f"which aggregate into <b>{f['module_edges']} distinct module-to-module "
        f"relationships</b> across {f.get('modules', 0)} modules. Each arrow below "
        f"means at least one file in the source module imports a file in the target.")
    if f.get("most_depended_on"):
        items = ", ".join(f"<b>{m}</b> ({n})" for m, n in f["most_depended_on"][:3])
        out.append(
            f"The most depended-upon modules are {items}. These are the "
            f"architectural foundations — a change to their public surface "
            f"propagates widest, so they warrant the most test coverage.")
    if f.get("most_dependent"):
        items = ", ".join(f"<b>{m}</b> ({n})" for m, n in f["most_dependent"][:3])
        out.append(
            f"The modules with the most outgoing dependencies are {items}. High "
            f"outgoing coupling usually marks orchestration or composition layers, "
            f"which are harder to unit test because they need many collaborators "
            f"substituted.")
    if f.get("cycles"):
        cyc = "; ".join(" → ".join(c) for c in f["cycles"][:3])
        out.append(
            f"<b>Circular dependencies detected:</b> {cyc}. Cycles between modules "
            f"prevent independent compilation and testing, and usually indicate a "
            f"missing abstraction that both sides should depend on instead.")
    else:
        out.append("No circular dependencies were found between modules at this "
                   "aggregation level, which is a good structural sign.")
    if f.get("truncated_to"):
        out.append(f"<i>The diagram shows the {f['truncated_to']} most connected "
                   f"modules; less connected ones are omitted for legibility.</i>")
    return out


def describe_functions(f):
    out = []
    if not f.get("cross_module"):
        out.append("No functions were found being called across module boundaries. "
                   "This can mean the modules are genuinely independent, or that the "
                   "scan could not resolve call sites for this language.")
        return out
    out.append(
        f"Scanning the source found <b>{f['total_functions']} named functions</b>, of "
        f"which <b>{f['cross_module']} are called from outside the module that defines "
        f"them</b>. Those are the de facto public API: the surface other parts of the "
        f"system rely on, regardless of what any access modifier says.")
    if f.get("top_shared"):
        top = f["top_shared"][:4]
        items = "; ".join(
            f"<b>{n}()</b> — defined in {home}, called from {ne} other module(s), "
            f"{tot} call sites" for n, ne, tot, home in top)
        out.append(f"The most widely shared functions are: {items}.")
        out.append(
            "Functions reaching this many modules are the highest-value test targets: "
            "a defect in one propagates everywhere it is called, and each additional "
            "caller is another behavior contract to preserve.")
    return out


def describe_keypoints(f, key_points):
    out = []
    if not f.get("key_points") or not f.get("vectors"):
        out.append("No capability landscape could be built — this view needs both "
                   "synthesized key points and file vectors. Run the <i>synth</i> and "
                   "<i>embed</i> stages.")
        return out
    out.append(
        f"The repository's behavior was synthesized into <b>{f['key_points']} key "
        f"points</b>, and every file carries a vector whose k-th dimension is that "
        f"file's similarity to key point k. Averaging those vectors per module gives "
        f"the heatmap below: which parts of the codebase are responsible for which "
        f"capability.")
    owners = f.get("owners", {})
    if owners:
        lines = []
        for k in sorted(owners)[:5]:
            mod, score = owners[k]
            text = key_points[k] if k < len(key_points) else f"Key point {k}"
            snippet = text if len(text) < 90 else text[:88] + "…"
            lines.append(f"<b>KP{k}</b> ({snippet}) → strongest in <b>{mod}</b> "
                         f"[{score:.2f}]")
        out.append("Capability ownership: " + "; ".join(lines) + ".")
        out.append(
            "A capability with one clearly dominant module is well encapsulated. One "
            "spread evenly across many modules is a cross-cutting concern — those are "
            "the ones where a requirement change forces edits in several places at "
            "once.")
    return out


def describe_responsibility(f):
    out = []
    if not f.get("total_functions"):
        out.append("No function metrics are available — run the <i>scan</i> stage to "
                   "populate per-file counts in PostgreSQL.")
        return out
    out.append(
        f"Across {f['modules']} modules the repository defines <b>{f['total_functions']} "
        f"functions and methods</b>. The chart shows how that behavior is distributed, "
        f"split by language so mixed-stack modules are visible.")
    if f.get("top_modules"):
        items = ", ".join(f"<b>{m}</b> ({n})" for m, n in f["top_modules"][:3])
        out.append(
            f"The largest concentrations of behavior are {items}. Modules carrying a "
            f"disproportionate share of the function count are natural candidates for "
            f"decomposition, and they are where test generation yields the most value "
            f"per file.")
    if f.get("most_shared_functions"):
        items = ", ".join(f"<b>{n}()</b> ({m} modules)"
                          for n, m in f["most_shared_functions"][:4])
        out.append(f"Functions whose responsibility is shared most widely: {items}.")
    return out


# ── Figures ───────────────────────────────────────────────────────────────────

def _save(fig, outdir: Path, name: str) -> Path:
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_module_dependencies(repo, edges, depth, max_nodes, outdir):
    """Directed graph of module → module imports, plus a findings dict."""
    mod_edges = Counter()
    for src, tgt in edges:
        a, b = module_of(src, depth), module_of(tgt, depth)
        if a != b:
            mod_edges[(a, b)] += 1

    findings = {"module_edges": len(mod_edges), "file_edges": len(edges)}

    if not mod_edges:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No cross-module dependencies recorded",
                ha="center", va="center", fontsize=13, color=C_MUTED)
        ax.axis("off")
        return _save(fig, outdir, "module_deps"), findings

    G = nx.DiGraph()
    for (a, b), w in mod_edges.items():
        G.add_edge(a, b, weight=w)

    if G.number_of_nodes() > max_nodes:
        deg = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
        keep = [n for n, _ in Counter(deg).most_common(max_nodes)]
        G = G.subgraph(keep).copy()
        findings["truncated_to"] = max_nodes

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.4, iterations=90, seed=42)
    in_deg = dict(G.in_degree())
    sizes = [700 + 320 * in_deg.get(n, 0) for n in G.nodes()]
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    mx = max(weights) if weights else 1
    widths = [0.6 + 3.0 * (w / mx) for w in weights]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE, width=widths,
                           arrows=True, arrowsize=13,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=C_PRIMARY, alpha=0.9, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5,
                            font_color="white", font_weight="bold")
    ax.set_title(f"Module dependency graph — {repo}\n"
                 f"node size = how many modules import it; "
                 f"edge width = number of file-level imports",
                 fontsize=11, pad=16)
    ax.axis("off")

    ranked_in = Counter()
    for (a, b), w in mod_edges.items():
        ranked_in[b] += w
    ranked_out = Counter()
    for (a, b), w in mod_edges.items():
        ranked_out[a] += w
    findings["most_depended_on"] = ranked_in.most_common(5)
    findings["most_dependent"]   = ranked_out.most_common(5)
    findings["modules"] = G.number_of_nodes()
    try:
        cycles = list(nx.simple_cycles(G))
        findings["cycles"] = [c for c in cycles if len(c) > 1][:5]
    except Exception:
        findings["cycles"] = []

    return _save(fig, outdir, "module_deps"), findings


def fig_function_relationships(repo, func_index, depth, max_funcs, outdir):
    """Bipartite-ish view: functions used across module boundaries."""
    cross = []
    for name, info in func_index.items():
        def_mods = {module_of(p, depth) for p in info["defs"]}
        call_mods = {module_of(p, depth) for p in info["calls"]}
        external = call_mods - def_mods
        if external:
            cross.append((name, len(external), sum(info["calls"].values()),
                          sorted(def_mods)[0], sorted(external)))

    findings = {"total_functions": len(func_index), "cross_module": len(cross)}

    if not cross:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No functions found crossing module boundaries",
                ha="center", va="center", fontsize=13, color=C_MUTED)
        ax.axis("off")
        return _save(fig, outdir, "func_rel"), findings

    cross.sort(key=lambda x: (-x[1], -x[2]))
    top = cross[:max_funcs]

    G = nx.DiGraph()
    fnodes, mnodes = set(), set()
    for name, n_ext, total, home, ext_mods in top:
        fn = f"{name}()"
        fnodes.add(fn)
        for m in ext_mods[:6]:
            mnodes.add(m)
            G.add_edge(m, fn)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.5 * len(G))))
    pos = nx.bipartite_layout(G, sorted(mnodes), align="vertical", scale=2.2)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE, width=1.0,
                           arrows=True, arrowsize=9)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(mnodes),
                           node_size=900, node_color=C_MUTED, linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sorted(fnodes),
                           node_size=800, node_color=C_ACCENT, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="black")
    ax.set_title(f"Cross-module function usage — {repo}\n"
                 f"grey = calling module, orange = function; "
                 f"an arrow means the module calls a function defined elsewhere",
                 fontsize=11, pad=16)
    ax.axis("off")

    findings["top_shared"] = [(n, ne, tot, home) for n, ne, tot, home, _ in top[:8]]
    return _save(fig, outdir, "func_rel"), findings


def fig_keypoint_landscape(repo, key_points, vectors, depth, outdir):
    """Heatmap: module (rows) vs key point (columns), mean alignment."""
    findings = {"key_points": len(key_points), "vectors": len(vectors)}
    if not key_points or not vectors:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No key points or vectors — run the synth and embed stages",
                ha="center", va="center", fontsize=13, color=C_MUTED)
        ax.axis("off")
        return _save(fig, outdir, "kp_landscape"), findings

    by_mod = defaultdict(list)
    for path, vec in vectors.items():
        by_mod[module_of(path, depth)].append(vec)

    mods = sorted(by_mod, key=lambda m: -len(by_mod[m]))[:18]
    n_kp = len(key_points)
    matrix = []
    for m in mods:
        vecs = by_mod[m]
        matrix.append([sum(v[k] for v in vecs) / len(vecs) if k < len(vecs[0]) else 0.0
                       for k in range(n_kp)])

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n_kp + 4), max(4, 0.4 * len(mods) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels([m if len(m) < 34 else "…" + m[-32:] for m in mods], fontsize=8)
    ax.set_xticks(range(n_kp))
    ax.set_xticklabels([f"KP{i}" for i in range(n_kp)], fontsize=8)
    ax.set_xlabel("Key point", fontsize=9)
    ax.set_title(f"Capability landscape — {repo}\n"
                 f"mean alignment of each module's files to each key point",
                 fontsize=11, pad=14)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean similarity")

    owner = {}
    for k in range(n_kp):
        best_i = max(range(len(mods)), key=lambda i: matrix[i][k])
        owner[k] = (mods[best_i], matrix[best_i][k])
    findings["owners"] = owner
    findings["modules_charted"] = len(mods)
    return _save(fig, outdir, "kp_landscape"), findings


def fig_module_responsibility(repo, files, func_index, depth, max_mods, outdir):
    """Stacked bars: how many functions each module defines, split by language."""
    per_mod_lang = defaultdict(Counter)
    per_mod_funcs = Counter()
    for f in files:
        m = module_of(f["path"], depth)
        n = (f.get("functions", 0) or 0) + (f.get("methods", 0) or 0)
        per_mod_funcs[m] += n
        per_mod_lang[m][f.get("language") or "Other"] += n

    findings = {"modules": len(per_mod_funcs),
                "total_functions": sum(per_mod_funcs.values())}

    if not per_mod_funcs:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No function metrics — run the scan stage",
                ha="center", va="center", fontsize=13, color=C_MUTED)
        ax.axis("off")
        return _save(fig, outdir, "mod_resp"), findings

    top = [m for m, _ in per_mod_funcs.most_common(max_mods)]
    langs = sorted({l for m in top for l in per_mod_lang[m]},
                   key=lambda l: -sum(per_mod_lang[m][l] for m in top))[:6]
    cmap = plt.get_cmap("tab20")
    colors = {l: cmap(i / max(len(langs) - 1, 1)) for i, l in enumerate(langs)}

    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(top) + 2)))
    left = [0] * len(top)
    for l in langs:
        vals = [per_mod_lang[m].get(l, 0) for m in top]
        ax.barh(range(len(top)), vals, left=left, label=l,
                color=colors[l], edgecolor="white", linewidth=0.5)
        left = [a + b for a, b in zip(left, vals)]

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([m if len(m) < 40 else "…" + m[-38:] for m in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Functions + methods defined", fontsize=9)
    ax.set_title(f"Module responsibility — {repo}\n"
                 f"how much behavior each module owns, by language",
                 fontsize=11, pad=14)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    findings["top_modules"] = per_mod_funcs.most_common(6)
    shared = sorted(
        ((n, len({module_of(p, depth) for p in v["calls"]}))
         for n, v in func_index.items()),
        key=lambda x: -x[1])[:6]
    findings["most_shared_functions"] = [s for s in shared if s[1] > 1]
    return _save(fig, outdir, "mod_resp"), findings


# ── Drill-down appendix (uses kb.diagram's image generators) ────────────────

def build_appendix_images(repo, repo_path, edges, findings, func_index, key_points,
                          imgdir, module_depth=2, max_items=6):
    """
    Generate actual PNG images for the drill-down details using diagram.py's
    image_* functions. Returns [(title, description, image_path)].

    Was three categories (dependency / function / capability detail) when
    diagram.py had function and keypoint diagram generators. It doesn't
    anymore, so:
      - dependency detail: unchanged, still image_deps. Module membership is
        checked via module_of(t, module_depth) == hot_mod — a raw string
        prefix check used to work here because the old module_of() always
        returned a literal prefix of the full path; it no longer does now
        that it strips the leading repo-name segment (see module_of).
      - function detail -> CLASS detail: shows the classes defined in the
        file that owns the most widely-shared function, via image_class
      - capability detail: dropped — there's no diagram.py equivalent left
        for a per-keypoint drill-down. The keypoint LANDSCAPE figure itself
        (figure 3 in the main report) is unaffected, since it never depended
        on kb.diagram — only this per-keypoint zoom-in did.
    """
    items = []

    # A. Hot file in the most depended-upon module
    mod_findings = findings.get("modules", {})
    if mod_findings.get("most_depended_on") and edges:
        hot_mod = mod_findings["most_depended_on"][0][0]
        incoming = Counter(t for _, t in edges if module_of(t, module_depth) == hot_mod)
        if incoming:
            hot_file, n = incoming.most_common(1)[0]
            try:
                img_path = imgdir / f"appendix_deps_{Path(hot_file).stem}.png"
                image_deps(repo, img_path, hot_file, depth=1)
                items.append((
                    f"A. Dependency detail — {short(hot_file, 3)}",
                    f"This is the single most imported file inside <b>{hot_mod}</b>, "
                    f"the module the rest of the system leans on most, with "
                    f"<b>{n} incoming file-level imports</b>. Its dependents are the "
                    f"blast radius of any change to it.",
                    img_path,
                ))
            except Exception as e:
                print(f"    (skipped dependency detail: {e})")

    # B. Classes in the file that owns the most widely shared function
    fn_findings = findings.get("functions", {})
    for name, n_mods, total, home in (fn_findings.get("top_shared") or [])[:2]:
        defs = func_index.get(name, {}).get("defs") or []
        if not defs or not repo_path:
            continue
        file_label = defs[0]
        try:
            # file_label is repo-name-prefixed (e.g. "flask/src/flask/app.py");
            # repo_path already ends in the repo name, so join against its
            # PARENT to avoid doubling it — same convention diagram.py uses.
            file_path = repo_path.parent / file_label
            img_path = imgdir / f"appendix_class_{Path(file_label).stem}.png"
            image_class(file_path, file_label, img_path)
            items.append((
                f"B. Class detail — {short(file_label, 3)}",
                f"<b>{name}()</b> is called from <b>{n_mods} other module(s)</b> "
                f"across {total} call sites — the highest cross-module reach found "
                f"in <b>{home}</b>. Below are the classes that file defines; a "
                f"function this widely relied on is a high-value test target "
                f"regardless of which class it lives on.",
                img_path,
            ))
        except Exception as e:
            print(f"    (skipped class detail for {name}: {e})")

    return items[:max_items]


# ── PDF ───────────────────────────────────────────────────────────────────────

def build_pdf(repo, out_path, figures, meta, appendix_images=None, repo_summary=None):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image, PageBreak, Table, TableStyle)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=22, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading1"], fontSize=15,
                        textColor=colors.HexColor(C_PRIMARY), spaceBefore=14, spaceAfter=8)
    h3 = ParagraphStyle("h3", parent=styles["Heading2"], fontSize=12,
                        textColor=colors.HexColor(C_ACCENT), spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10,
                          leading=15, spaceAfter=8)
    cap = ParagraphStyle("cap", parent=styles["Normal"], fontSize=8.5,
                         textColor=colors.HexColor("#5a6570"), spaceBefore=4, spaceAfter=12)

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                            leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                            title=f"Groundwork System Report — {repo}")
    story = []

    story.append(Paragraph(f"System Report", h1))
    story.append(Paragraph(f"Repository: <b>{repo}</b>", body))
    story.append(Spacer(1, 10))

    # Repository Summary
    if repo_summary:
        story.append(Paragraph("Repository Overview", h2))
        
        # Summary table
        cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=9, leading=12)
        cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white,
                               fontName="Helvetica-Bold")
        rows = [[Paragraph("Attribute", cellb), Paragraph("Value", cellb)]]
        
        if repo_summary.get("primary_language"):
            rows.append([Paragraph("Primary Language", cell), 
                        Paragraph(repo_summary["primary_language"], cell)])
        if repo_summary.get("file_count"):
            rows.append([Paragraph("Total Files", cell), 
                        Paragraph(str(repo_summary["file_count"]), cell)])
        if repo_summary.get("line_count"):
            rows.append([Paragraph("Total Lines", cell), 
                        Paragraph(f"{repo_summary['line_count']:,}", cell)])
        
        # Main modules
        if repo_summary.get("main_modules"):
            mods = ", ".join(repo_summary["main_modules"][:5])
            rows.append([Paragraph("Main Modules", cell), 
                        Paragraph(mods, cell)])
        
        t = Table(rows, colWidths=[1.7 * inch, 4.6 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))
        
        # Description
        if repo_summary.get("description"):
            story.append(Paragraph(f"<b>Description:</b> {repo_summary['description']}", body))
            story.append(Spacer(1, 6))
        
        # Key points
        if repo_summary.get("key_points"):
            story.append(Paragraph("<b>Key Capabilities:</b>", body))
            for i, kp in enumerate(repo_summary["key_points"][:5], 1):
                story.append(Paragraph(f"{i}. {kp[:150]}{'...' if len(kp) > 150 else ''}", body))
            story.append(Spacer(1, 8))
        
        # Data sources summary
        story.append(Paragraph("<b>Knowledge Sources:</b>", body))
        cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=9, leading=12)
        rows = [[Paragraph("Source", cellb), Paragraph("Content", cellb)]]
        for a, b in meta["summary_rows"]:
            rows.append([Paragraph(a, cell), Paragraph(b, cell)])
        t = Table(rows, colWidths=[1.7 * inch, 4.6 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
    
    story.append(Paragraph(meta["intro"], body))

    # Main figures
    for i, (title, img, paragraphs, caption) in enumerate(figures):
        story.append(PageBreak())
        story.append(Paragraph(title, h2))
        for p in paragraphs:
            story.append(Paragraph(p, body))
        if img and Path(img).exists():
            from PIL import Image as PILImage
            with PILImage.open(img) as im:
                w, h = im.size
            max_w = 6.3 * inch
            max_h = 6.6 * inch
            scale = min(max_w / w, max_h / h)
            story.append(Spacer(1, 6))
            story.append(Image(str(img), width=w * scale, height=h * scale))
            story.append(Paragraph(caption, cap))

    # Appendix with actual images
    if appendix_images:
        story.append(PageBreak())
        story.append(Paragraph("Appendix — Drill-down diagrams", h2))
        story.append(Paragraph(
            "The views above describe the system as a whole. The diagrams here zoom "
            "into the specific files, functions, and capabilities those views "
            "identified as most significant.",
            body))

        for title, desc, img_path in appendix_images:
            story.append(Spacer(1, 10))
            story.append(Paragraph(title, h3))
            story.append(Paragraph(desc, body))
            if img_path and Path(img_path).exists():
                from PIL import Image as PILImage
                with PILImage.open(img_path) as im:
                    w, h = im.size
                max_w = 6.0 * inch
                max_h = 5.0 * inch
                scale = min(max_w / w, max_h / h)
                story.append(Spacer(1, 4))
                story.append(Image(str(img_path), width=w * scale, height=h * scale))

    doc.build(story)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a repository-wide system report (diagrams + PDF)")
    ap.add_argument("--repo", required=True, help="Repository name")
    ap.add_argument("--repo-path", default=None, help="Source location on disk")
    ap.add_argument("--out", default=None, help="Output PDF (default: reports/<repo>_system_report.pdf)")
    ap.add_argument("--module-depth", type=int, default=2,
                    help="Path segments that define a module (default: 2)")
    ap.add_argument("--max-nodes", type=int, default=30,
                    help="Max modules drawn in the dependency graph (default: 30)")
    ap.add_argument("--max-funcs", type=int, default=14,
                    help="Max functions in the relationship view (default: 14)")
    ap.add_argument("--scan-files", type=int, default=600,
                    help="Max source files to scan for functions (default: 600)")
    ap.add_argument("--appendix-items", type=int, default=6,
                    help="Drill-down Mermaid diagrams to append (0 disables)")
    ap.add_argument("--keep-images", action="store_true",
                    help="Keep the generated PNGs next to the PDF")
    args = ap.parse_args()

    repo = args.repo
    repo_path = Path(args.repo_path) if args.repo_path else None
    if repo_path is None:
        for cand in (Path("./repos") / repo, Path(repo)):
            if cand.is_dir():
                repo_path = cand
                break

    out_path = Path(args.out) if args.out else Path("reports") / f"{repo}_system_report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgdir = out_path.parent / f"{repo}_figures" if args.keep_images \
             else Path(tempfile.mkdtemp(prefix="gw_figs_"))
    imgdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Repository : {repo}")
    print(f"  Source     : {repo_path if repo_path and repo_path.is_dir() else '(not found — function views limited)'}")
    print(f"  Output     : {out_path}\n")

    print("  Loading knowledge base...")
    edges      = fetch_edges(repo)
    files      = fetch_files(repo)
    key_points = fetch_key_points(repo)
    vectors    = fetch_vectors(repo)
    print(f"    {len(files)} files, {len(edges)} import edges, "
          f"{len(key_points)} key points, {len(vectors)} vectors")

    func_index = {}
    if repo_path and repo_path.is_dir() and files:
        print(f"  Scanning up to {args.scan_files} source files for functions...")
        func_index = scan_repo_functions(repo_path, repo, files, args.scan_files)
        print(f"    {len(func_index)} functions indexed")

    print("  Rendering figures...")
    p1, f1 = fig_module_dependencies(repo, edges, args.module_depth, args.max_nodes, imgdir)
    p2, f2 = fig_function_relationships(repo, func_index, args.module_depth, args.max_funcs, imgdir)
    p3, f3 = fig_keypoint_landscape(repo, key_points, vectors, args.module_depth, imgdir)
    p4, f4 = fig_module_responsibility(repo, files, func_index, args.module_depth, args.max_nodes, imgdir)

    # Gather repository summary
    print("  Gathering repository summary...")
    repo_summary = fetch_repo_metadata(repo, repo_path if repo_path else Path("."))

    langs = Counter(f.get("language") or "Other" for f in files)
    lang_str = ", ".join(f"{l} ({n})" for l, n in langs.most_common(5)) or "—"

    meta = {
        "summary_rows": [
            ["PostgreSQL", f"{len(files)} files — {lang_str}"],
            ["Kùzu", f"{len(edges)} DEPENDS_ON edges between files"],
            ["ChromaDB", f"{len(vectors)} file vectors across {len(key_points)} key points"],
            ["Source scan", f"{len(func_index)} functions indexed from disk"],
        ],
        "intro": (
            "This report describes the repository as a system rather than as "
            "individual files. Each view aggregates to <b>module</b> level "
            f"(the first {args.module_depth} path segments), because at repository "
            "scale a file-level graph is too dense to read. Every figure is followed "
            "by findings computed from the data, not generic commentary."),
    }

    figures = [
        ("1. Module dependency graph", p1, describe_module_deps(f1),
         "Directed graph of module-to-module imports, aggregated from file-level "
         "DEPENDS_ON edges in the graph store."),
        ("2. Cross-module function relationships", p2, describe_functions(f2),
         "Functions defined in one module and called from others, found by scanning "
         "the source; the knowledge base indexes dependencies at file level only."),
        ("3. Capability landscape", p3, describe_keypoints(f3, key_points),
         "Mean similarity of each module's files to each synthesized key point, "
         "computed from the ChromaDB vectors."),
        ("4. Module responsibility", p4, describe_responsibility(f4),
         "Functions and methods defined per module, split by language, from the "
         "PostgreSQL file metrics."),
    ]

    print("  Building drill-down appendix images...")
    appendix_images = build_appendix_images(
        repo, repo_path or Path('.'), edges,
        {"modules": f1, "functions": f2, "keypoints": f3, "responsibility": f4},
        func_index, key_points, imgdir,
        module_depth=args.module_depth, max_items=args.appendix_items,
    ) if args.appendix_items > 0 else []
    if appendix_images:
        print(f"    {len(appendix_images)} detail diagram(s)")

    print("  Building PDF...")
    build_pdf(repo, out_path, figures, meta, appendix_images, repo_summary)
    print(f"\n  ✓ Wrote {out_path}")
    if args.keep_images:
        print(f"  ✓ Figures kept in {imgdir}")
    print()


if __name__ == "__main__":
    main()