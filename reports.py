#!/usr/bin/env python3
"""
Groundwork — Knowledge Base Reports

Three report types, all written to PDF. Each builds on existing Groundwork
modules rather than re-reading the stores its own way:

    kb.system_report   store fetchers (fetch_edges/files/key_points/vectors),
                       module_of, figure saving
    kb.diagram         multi-language source scanning, path helpers
    kb.report_pdf      shared ReportLab styling
    kb.grab_context    the repo-level inference prompt

  overview     What the repository is, what it does, and which files implement
               each capability. One section per key point.

  file         A single file in context: its functions, what it imports, what
               imports it, its business rules, and which capabilities it serves.

  capability   A layered flow diagram — repository → capabilities → files —
               showing every file behind each capability and how the
               capabilities compose into the system.

Run from the PROJECT ROOT.

Usage:
    python3 -m kb.reports --repo nopCommerce --type overview
    python3 -m kb.reports --repo nopCommerce --type file --file Libraries/Nop.Core/Svc0.cs
    python3 -m kb.reports --repo nopCommerce --type capability
    python3 -m kb.reports --repo nopCommerce --type overview --no-llm

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (Paragraph, Spacer, Image, PageBreak,
                                Table, TableStyle, KeepTogether)

from dotenv import load_dotenv

# Reuse the store access and helpers that already exist
from system_report import (
    module_of, fetch_edges, fetch_files, fetch_key_points, fetch_vectors,
    _save, C_PRIMARY, C_ACCENT, C_MUTED, C_EDGE,
)
from kb.diagram import (
    SCANNABLE, _strip_comments, short, resolve_repo_path,
)
from report_pdf import _styles, _doc, _inline, timestamp

load_dotenv()

MAX_FILES_PER_CAPABILITY = 12
OWNERSHIP_MARGIN = 0.08          # how much a capability must lead by to "own" a file


# ── Shared analysis ───────────────────────────────────────────────────────────

def capability_map(key_points, vectors, top_n=MAX_FILES_PER_CAPABILITY):
    """
    {kp_index: [(file, score), ...]} — the files most aligned to each capability,
    ranked. A file can appear under several capabilities; that is meaningful, it
    means the file serves more than one.
    """
    out = {}
    for k in range(len(key_points)):
        ranked = sorted(((v[k], f) for f, v in vectors.items() if k < len(v)),
                        reverse=True)
        out[k] = [(f, s) for s, f in ranked[:top_n]]
    return out


def primary_capability(vectors):
    """{file: (kp_index, score)} — the single capability each file serves most."""
    out = {}
    for f, v in vectors.items():
        if not v:
            continue
        k = max(range(len(v)), key=lambda i: v[i])
        out[f] = (k, v[k])
    return out


def scan_file_functions(repo_path: Path, rel_file: str):
    """
    (functions_defined, imports_found) for one file, using kb.diagram's language
    table and comment stripping so the rules stay in one place.
    """
    ext = Path(rel_file).suffix.lower()
    full = repo_path / rel_file
    if ext not in SCANNABLE or not full.is_file():
        return [], []
    try:
        raw = full.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], []
    text = _strip_comments(raw, SCANNABLE[ext])

    defs, imports = [], []
    if ext == ".py":
        defs = re.findall(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", text, re.M)
        imports = [m for m in re.findall(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))",
                                         text, re.M)]
        imports = [a or b for a, b in imports]
    elif ext in (".cs", ".cshtml", ".razor"):
        defs = re.findall(
            r"(?:public|private|protected|internal)\s+"
            r"(?:static\s+|virtual\s+|override\s+|async\s+|sealed\s+|partial\s+)*"
            r"[\w<>\[\],\.\?]+\s+(\w+)\s*(?:<[^>]*>)?\s*\(", text)
        imports = re.findall(r"^\s*using\s+([\w.]+)\s*;", text, re.M)
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        defs = re.findall(r"function\s+(\w+)\s*\(", text)
        defs += re.findall(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", text)
        imports = re.findall(r"""from\s+['"]([^'"]+)['"]""", text)
        imports += re.findall(r"""require\s*\(\s*['"]([^'"]+)['"]""", text)

    skip = {"if", "for", "while", "switch", "catch", "using", "lock", "return", "get", "set"}
    defs = [d for d in dict.fromkeys(defs) if d not in skip and not d.startswith("_")]
    return defs, list(dict.fromkeys(imports))


def infer_purpose(repo, key_points, files, use_llm=True):
    """
    A prose statement of what the repository is. Uses the same architect prompt
    grab_context uses for repo-level questions; falls back to a derived summary
    when the LLM is unavailable.
    """
    langs = Counter(f.get("language") or "Other" for f in files)
    lang_str = ", ".join(f"{l} ({n} files)" for l, n in langs.most_common(4))
    total_lines = sum(f.get("lines", 0) for f in files)

    if use_llm and key_points and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            from kb.grab_context import REPO_LEVEL_SYSTEM
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            profile = (f"Repository: {repo}\n"
                       f"Files: {len(files)}, lines: {total_lines}\n"
                       f"Languages: {lang_str}\n"
                       f"Top-level layout: "
                       f"{', '.join(m for m, _ in Counter(module_of(f['path'], 2) for f in files).most_common(8))}")
            catalog = "\n".join(f"- {kp}" for kp in key_points)
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": REPO_LEVEL_SYSTEM},
                    {"role": "user", "content":
                        f"STRUCTURAL PROFILE:\n{profile}\n\n"
                        f"CAPABILITY CATALOG:\n{catalog}\n\n"
                        f"In 2-3 paragraphs, explain what this codebase is, its domain, "
                        f"and its likely architecture. Justify from the evidence."},
                ],
            )
            return resp.choices[0].message.content.strip(), True
        except Exception as e:
            print(f"    (LLM inference unavailable: {e})")

    derived = (
        f"{repo} contains {len(files)} indexed files totalling {total_lines:,} lines, "
        f"written predominantly in {lang_str or 'an unrecorded language'}. "
        f"The knowledge base synthesized {len(key_points)} distinct capabilities from "
        f"the business rules extracted across those files; each is detailed below "
        f"together with the files that implement it.")
    return derived, False


# ── Report 1: repository overview ─────────────────────────────────────────────

def fig_capability_sizes(repo, key_points, vectors, outdir):
    """Horizontal bars: how many files each capability primarily owns."""
    prim = primary_capability(vectors)
    counts = Counter(k for k, _ in prim.values())
    fig, ax = plt.subplots(figsize=(9, max(2.4, 0.42 * len(key_points) + 1.2)))
    idx = list(range(len(key_points)))
    vals = [counts.get(k, 0) for k in idx]
    ax.barh(idx, vals, color=C_PRIMARY, edgecolor="white")
    ax.set_yticks(idx)
    ax.set_yticklabels(
        [f"KP{k}  " + (key_points[k][:52] + "…" if len(key_points[k]) > 52 else key_points[k])
         for k in idx], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Files whose strongest alignment is this capability", fontsize=9)
    ax.set_title(f"Capability footprint — {repo}", fontsize=11, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, outdir, "capability_sizes"), counts


def build_overview(repo, files, key_points, vectors, edges, outdir, use_llm=True):
    st = _styles()
    story = []

    story.append(Paragraph("Repository Overview", st["title"]))
    story.append(Paragraph(f"{repo}  ·  {timestamp()}", st["sub"]))

    # Purpose
    story.append(Paragraph("What this repository is", st["h1"]))
    purpose, from_llm = infer_purpose(repo, key_points, files, use_llm)
    for para in purpose.split("\n\n"):
        if para.strip():
            story.append(Paragraph(_inline(para.strip()), st["body"]))
    if not from_llm and key_points:
        story.append(Paragraph(
            "<i>This summary is derived from the stored metrics. Set OPENAI_API_KEY "
            "for a synthesized architectural reading.</i>", st["sub"]))

    # Structural profile
    langs = Counter(f.get("language") or "Other" for f in files)
    mods = Counter(module_of(f["path"], 2) for f in files)
    rows = [[Paragraph("Property", st["cellh"]), Paragraph("Value", st["cellh"])]]
    for label, val in [
        ("Indexed files", f"{len(files):,}"),
        ("Lines of code", f"{sum(f.get('lines', 0) for f in files):,}"),
        ("Classes / functions / methods",
         f"{sum(f.get('classes',0) for f in files):,} / "
         f"{sum(f.get('functions',0) for f in files):,} / "
         f"{sum(f.get('methods',0) for f in files):,}"),
        ("Languages", ", ".join(f"{l} ({n})" for l, n in langs.most_common(6)) or "—"),
        ("Top-level modules", ", ".join(m for m, _ in mods.most_common(6)) or "—"),
        ("Import edges", f"{len(edges):,}"),
        ("Capabilities", f"{len(key_points)}"),
        ("Files with vectors", f"{len(vectors):,}"),
    ]:
        rows.append([Paragraph(label, st["cell"]), Paragraph(val, st["cell"])])
    t = Table(rows, colWidths=[2.0 * inch, 4.3 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Paragraph("Structural profile", st["h1"]))
    story.append(t)

    # Capability summary
    story.append(PageBreak())
    story.append(Paragraph("Capability summary", st["h1"]))
    if not key_points:
        story.append(Paragraph(
            "No capabilities have been synthesized. Run the <i>synth</i> stage:<br/>"
            "<font face='Courier' size='8'>python3 -m kb.vector.synthesize --repo "
            f"{repo}</font>", st["body"]))
    else:
        img, counts = fig_capability_sizes(repo, key_points, vectors, outdir)
        story.append(Paragraph(
            f"The knowledge base synthesized <b>{len(key_points)} capabilities</b> from "
            f"the business rules across {len(files)} files. The chart shows how many "
            f"files each capability primarily owns — a file is counted under the "
            f"capability it aligns to most strongly.", st["body"]))
        if img and Path(img).exists():
            from PIL import Image as PILImage
            with PILImage.open(img) as im:
                w, h = im.size
            scale = min(6.3 * inch / w, 3.6 * inch / h)
            story.append(Image(str(img), width=w * scale, height=h * scale))

        big = counts.most_common(1)
        small = [k for k in range(len(key_points)) if counts.get(k, 0) == 0]
        if big:
            k, n = big[0]
            story.append(Paragraph(
                f"<b>KP{k}</b> is the largest capability, primarily served by {n} files. "
                f"Capabilities owning many files are the system's centre of gravity; "
                f"they carry the most behavior and warrant the deepest testing.",
                st["body"]))
        if small:
            story.append(Paragraph(
                f"Capabilities <b>{', '.join('KP'+str(k) for k in small)}</b> are not the "
                f"strongest match for any file. That usually means the behavior is "
                f"spread thinly across files that serve something else more strongly — "
                f"a cross-cutting concern rather than a dedicated module.", st["body"]))

    # One section per key point
    cap_map = capability_map(key_points, vectors)
    for k, kp in enumerate(key_points):
        story.append(PageBreak())
        story.append(Paragraph(f"Key Point {k}", st["h1"]))
        story.append(Paragraph(_inline(kp), st["body"]))

        entries = cap_map.get(k, [])
        if not entries:
            story.append(Paragraph(
                "No files carry a vector for this capability — run the "
                "<i>embed</i> stage.", st["body"]))
            continue

        mods_here = Counter(module_of(f, 2) for f, _ in entries)
        owner, owner_n = mods_here.most_common(1)[0]
        story.append(Paragraph(
            f"The {len(entries)} most aligned files sit mainly in <b>{owner}</b> "
            f"({owner_n} of them). Alignment is the file's similarity to this "
            f"capability, computed from its business rules.", st["body"]))

        rows = [[Paragraph("File", st["cellh"]), Paragraph("Module", st["cellh"]),
                 Paragraph("Alignment", st["cellh"])]]
        for f, s in entries:
            rows.append([Paragraph(short(f, 3), st["cell"]),
                         Paragraph(module_of(f, 2), st["cell"]),
                         Paragraph(f"{s:.3f}", st["cell"])])
        t = Table(rows, colWidths=[2.9 * inch, 2.3 * inch, 0.9 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f6f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
    return story


# ── Report 2: file summary ────────────────────────────────────────────────────

def build_file_report(repo, rel_file, repo_path, files, key_points, vectors,
                      edges, outdir):
    st = _styles()
    story = []

    meta = next((f for f in files if f["path"] == rel_file), None)
    story.append(Paragraph("File Report", st["title"]))
    story.append(Paragraph(f"{rel_file}  ·  {repo}  ·  {timestamp()}", st["sub"]))

    if meta is None:
        story.append(Paragraph(
            f"<b>{rel_file}</b> is not in the knowledge base for {repo}. Check the "
            f"path matches how it is stored, or run the <i>scan</i> stage.", st["body"]))
        return story

    # Identity
    rows = [[Paragraph("Property", st["cellh"]), Paragraph("Value", st["cellh"])]]
    for label, val in [
        ("Module", module_of(rel_file, 2)),
        ("Language", str(meta.get("language"))),
        ("Lines", f"{meta.get('lines', 0):,}"),
        ("Classes", str(meta.get("classes", 0))),
        ("Functions", str(meta.get("functions", 0))),
        ("Methods", str(meta.get("methods", 0))),
    ]:
        rows.append([Paragraph(label, st["cell"]), Paragraph(val, st["cell"])])
    t = Table(rows, colWidths=[1.6 * inch, 4.7 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # Functions and imports, from the source
    defs, imports = ([], [])
    if repo_path and repo_path.is_dir():
        defs, imports = scan_file_functions(repo_path, rel_file)

    story.append(Paragraph("Functions defined", st["h1"]))
    if defs:
        story.append(Paragraph(
            f"This file defines <b>{len(defs)}</b> named function(s) or method(s), "
            f"scanned from the source:", st["body"]))
        story.append(Paragraph(
            ", ".join(f"<font face='Courier' size='8.6'>{d}()</font>" for d in defs[:60]),
            st["body"]))
    else:
        story.append(Paragraph(
            "No function definitions were scanned. Either the source is not on disk "
            "at the expected path, or this language is not in the scanner's table.",
            st["body"]))

    story.append(Paragraph("Imports", st["h1"]))
    if imports:
        story.append(Paragraph(
            f"The file declares <b>{len(imports)}</b> import(s):", st["body"]))
        story.append(Paragraph(
            ", ".join(f"<font face='Courier' size='8.6'>{i}</font>" for i in imports[:50]),
            st["body"]))
    else:
        story.append(Paragraph("No imports were found in the source.", st["body"]))

    # Relationships from the graph
    story.append(Paragraph("How it relates to the rest of the codebase", st["h1"]))
    out_deps = sorted({t for s, t in edges if s == rel_file})
    in_deps  = sorted({s for s, t in edges if t == rel_file})

    story.append(Paragraph(
        f"The graph records <b>{len(out_deps)} outgoing</b> and "
        f"<b>{len(in_deps)} incoming</b> dependency edge(s) for this file. "
        f"Outgoing edges are what it relies on; incoming edges are the blast radius "
        f"of changing it.", st["body"]))

    def dep_table(title, items, note):
        if not items:
            return [Paragraph(title, st["h2"]),
                    Paragraph(note, st["body"])]
        rows = [[Paragraph("File", st["cellh"]), Paragraph("Module", st["cellh"])]]
        for d in items[:25]:
            rows.append([Paragraph(short(d, 3), st["cell"]),
                         Paragraph(module_of(d, 2), st["cell"])])
        tt = Table(rows, colWidths=[3.4 * inch, 2.9 * inch])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f6f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return [Paragraph(title, st["h2"]), tt]

    story += dep_table("Depends on", out_deps,
                       "No outgoing dependencies recorded.")
    story += dep_table("Depended on by", in_deps,
                       "Nothing in the repository imports this file, so it is either "
                       "an entry point, dead code, or its importers use a form the "
                       "resolver could not match.")

    # Local neighbourhood diagram
    if out_deps or in_deps:
        img = fig_file_neighbourhood(repo, rel_file, out_deps, in_deps, outdir)
        if img and Path(img).exists():
            from PIL import Image as PILImage
            with PILImage.open(img) as im:
                w, h = im.size
            scale = min(6.3 * inch / w, 4.2 * inch / h)
            story.append(Spacer(1, 8))
            story.append(Image(str(img), width=w * scale, height=h * scale))

    # Capabilities served
    story.append(PageBreak())
    story.append(Paragraph("Capabilities this file serves", st["h1"]))
    vec = vectors.get(rel_file)
    if not vec or not key_points:
        story.append(Paragraph(
            "No capability vector for this file — run the <i>embed</i> stage.", st["body"]))
    else:
        ranked = sorted(((s, k) for k, s in enumerate(vec)), reverse=True)
        rows = [[Paragraph("Capability", st["cellh"]),
                 Paragraph("Alignment", st["cellh"])]]
        for s, k in ranked[:6]:
            text = key_points[k] if k < len(key_points) else f"Key point {k}"
            rows.append([Paragraph(f"<b>KP{k}</b> — {_inline(text)}", st["cell"]),
                         Paragraph(f"{s:.3f}", st["cell"])])
        tt = Table(rows, colWidths=[5.4 * inch, 0.9 * inch])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f6f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tt)
        top_s, top_k = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0
        if top_s - second > OWNERSHIP_MARGIN:
            story.append(Paragraph(
                f"This file is clearly dedicated to <b>KP{top_k}</b> — it leads the "
                f"next capability by {top_s - second:.2f}. Focused files like this are "
                f"the easiest to test, because their expected behavior is unambiguous.",
                st["body"]))
        else:
            story.append(Paragraph(
                f"This file serves several capabilities at similar strength "
                f"(top two within {top_s - second:.2f}). Files spanning capabilities "
                f"often mix concerns and are worth reviewing for a split.", st["body"]))
    return story


def fig_file_neighbourhood(repo, rel_file, out_deps, in_deps, outdir):
    """Local dependency neighbourhood, one hop each way."""
    G = nx.DiGraph()
    centre = short(rel_file, 2)
    G.add_node(centre)
    for d in in_deps[:12]:
        G.add_edge(short(d, 2), centre)
    for d in out_deps[:12]:
        G.add_edge(centre, short(d, 2))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    layers = {}
    for n in G.nodes():
        layers[n] = 1 if n == centre else (0 if G.has_edge(n, centre) else 2)
    nx.set_node_attributes(G, layers, "layer")
    try:
        pos = nx.multipartite_layout(G, subset_key="layer")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE, width=1.2,
                           arrows=True, arrowsize=12)
    others = [n for n in G.nodes() if n != centre]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=others, node_size=1400,
                           node_color=C_MUTED, linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[centre], node_size=2200,
                           node_color=C_ACCENT, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white")
    ax.set_title(f"Dependency neighbourhood — {short(rel_file, 2)}\n"
                 f"left: files that import it   ·   right: files it imports",
                 fontsize=10, pad=14)
    ax.axis("off")
    return _save(fig, outdir, "file_neighbourhood")


# ── Report 3: capability flow ─────────────────────────────────────────────────

def fig_capability_flow(repo, key_points, vectors, files_per_cap, outdir):
    """
    Layered flow: repository → capabilities → the files behind each.
    Files serving more than one capability get an edge to each, which is what
    makes the cross-cutting structure visible.
    """
    cap_map = capability_map(key_points, vectors, top_n=files_per_cap)
    G = nx.DiGraph()
    root = repo
    G.add_node(root, layer=0)

    for k in range(len(key_points)):
        cap = f"KP{k}"
        G.add_node(cap, layer=1)
        G.add_edge(root, cap)
        for f, s in cap_map.get(k, []):
            leaf = short(f, 2)
            G.add_node(leaf, layer=2)
            G.add_edge(cap, leaf, weight=s)

    if G.number_of_nodes() <= 1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No capabilities or vectors — run synth and embed",
                ha="center", va="center", fontsize=13, color=C_MUTED)
        ax.axis("off")
        return _save(fig, outdir, "capability_flow"), {}

    n_leaves = sum(1 for n, d in G.nodes(data=True) if d.get("layer") == 2)
    fig, ax = plt.subplots(figsize=(15, max(7, 0.24 * n_leaves + 3)))
    pos = nx.multipartite_layout(G, subset_key="layer", scale=3.0)

    caps = [n for n, d in G.nodes(data=True) if d.get("layer") == 1]
    leaves = [n for n, d in G.nodes(data=True) if d.get("layer") == 2]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE, width=0.9,
                           arrows=True, arrowsize=8, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[root], node_size=3200,
                           node_color=C_PRIMARY, linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=caps, node_size=1700,
                           node_color=C_ACCENT, linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=leaves, node_size=520,
                           node_color=C_MUTED, alpha=0.85, linewidths=0)

    nx.draw_networkx_labels(G, pos, ax=ax, labels={root: root},
                            font_size=10, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={c: c for c in caps},
                            font_size=8.5, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={l: l for l in leaves},
                            font_size=5.6, font_color="black")

    ax.set_title(f"Capability flow — {repo}\n"
                 f"repository → capabilities → implementing files; "
                 f"a file linked to several capabilities serves all of them",
                 fontsize=12, pad=18)
    ax.axis("off")

    # Which files serve more than one capability
    multi = Counter()
    for k in range(len(key_points)):
        for f, _ in cap_map.get(k, []):
            multi[f] += 1
    findings = {
        "capabilities": len(key_points),
        "files_charted": len(leaves),
        "cross_cutting": [(f, n) for f, n in multi.most_common(8) if n > 1],
        "cap_map": cap_map,
    }
    return _save(fig, outdir, "capability_flow"), findings


def fig_capability_overlap(repo, key_points, vectors, outdir):
    """Capability-to-capability overlap: shared files between capabilities."""
    cap_map = capability_map(key_points, vectors)
    sets = {k: {f for f, _ in v} for k, v in cap_map.items()}
    G = nx.Graph()
    for k in range(len(key_points)):
        G.add_node(f"KP{k}")
    pairs = []
    for a in range(len(key_points)):
        for b in range(a + 1, len(key_points)):
            shared = len(sets.get(a, set()) & sets.get(b, set()))
            if shared:
                G.add_edge(f"KP{a}", f"KP{b}", weight=shared)
                pairs.append((f"KP{a}", f"KP{b}", shared))

    fig, ax = plt.subplots(figsize=(9, 6.5))
    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "Capabilities share no files — cleanly separated",
                ha="center", va="center", fontsize=12, color=C_MUTED)
        ax.axis("off")
        return _save(fig, outdir, "capability_overlap"), []

    pos = nx.circular_layout(G)
    w = [G[u][v]["weight"] for u, v in G.edges()]
    mx = max(w) or 1
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_EDGE,
                           width=[0.6 + 4.0 * (x / mx) for x in w])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500,
                           node_color=C_ACCENT, linewidths=0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, font_size=7,
        edge_labels={(u, v): str(G[u][v]["weight"]) for u, v in G.edges()})
    ax.set_title(f"Capability overlap — {repo}\n"
                 f"edge weight = files appearing in both capabilities' top files",
                 fontsize=11, pad=14)
    ax.axis("off")
    return _save(fig, outdir, "capability_overlap"), sorted(
        pairs, key=lambda p: -p[2])[:6]


def build_capability_report(repo, key_points, vectors, files, outdir, files_per_cap):
    st = _styles()
    story = []
    story.append(Paragraph("Capability System Diagram", st["title"]))
    story.append(Paragraph(f"{repo}  ·  {timestamp()}", st["sub"]))

    if not key_points or not vectors:
        story.append(Paragraph(
            "This report needs both synthesized key points and file vectors. Run:<br/>"
            f"<font face='Courier' size='8'>python3 -m kb.vector.synthesize --repo {repo}"
            f"<br/>python3 -m kb.vector.embeddings --repo {repo}</font>", st["body"]))
        return story

    flow_img, flow = fig_capability_flow(repo, key_points, vectors, files_per_cap, outdir)

    story.append(Paragraph("How capabilities compose the system", st["h1"]))
    story.append(Paragraph(
        f"The diagram traces the repository down through its <b>{flow['capabilities']} "
        f"capabilities</b> to the <b>{flow['files_charted']} files</b> that implement "
        f"them. Reading left to right: the repository is the sum of its capabilities, "
        f"and each capability is realized by the files linked beneath it. A file "
        f"appearing under more than one capability serves all of them.", st["body"]))

    if flow_img and Path(flow_img).exists():
        from PIL import Image as PILImage
        with PILImage.open(flow_img) as im:
            w, h = im.size
        scale = min(6.4 * inch / w, 7.6 * inch / h)
        story.append(Image(str(flow_img), width=w * scale, height=h * scale))

    if flow.get("cross_cutting"):
        story.append(PageBreak())
        story.append(Paragraph("Cross-cutting files", st["h1"]))
        items = ", ".join(f"<b>{short(f, 2)}</b> ({n} capabilities)"
                          for f, n in flow["cross_cutting"][:6])
        story.append(Paragraph(
            f"These files appear under multiple capabilities: {items}. A file serving "
            f"several capabilities is either a genuine shared utility or a sign that "
            f"concerns have been mixed. Either way they carry outsized risk — a change "
            f"affects more than one area of behavior at once, so they deserve the most "
            f"thorough tests.", st["body"]))
    else:
        story.append(Paragraph(
            "No file appears under more than one capability, meaning the capabilities "
            "are cleanly separated across the codebase.", st["body"]))

    # Overlap view
    ov_img, pairs = fig_capability_overlap(repo, key_points, vectors, outdir)
    story.append(Paragraph("How capabilities relate to each other", st["h1"]))
    if pairs:
        top = pairs[0]
        story.append(Paragraph(
            f"Capabilities are connected when they draw on the same files. The "
            f"strongest coupling is between <b>{top[0]}</b> and <b>{top[1]}</b>, "
            f"sharing {top[2]} file(s). Tightly coupled capabilities tend to change "
            f"together, so a requirement touching one usually touches the other.",
            st["body"]))
    else:
        story.append(Paragraph(
            "The capabilities share no files at this ranking depth — each is "
            "implemented by a distinct set of files.", st["body"]))
    if ov_img and Path(ov_img).exists():
        from PIL import Image as PILImage
        with PILImage.open(ov_img) as im:
            w, h = im.size
        scale = min(5.4 * inch / w, 4.4 * inch / h)
        story.append(Image(str(ov_img), width=w * scale, height=h * scale))

    # Per-capability file listing
    cap_map = flow.get("cap_map", {})
    for k, kp in enumerate(key_points):
        entries = cap_map.get(k, [])
        if not entries:
            continue
        story.append(PageBreak())
        story.append(Paragraph(f"KP{k} — implementing files", st["h1"]))
        story.append(Paragraph(_inline(kp), st["body"]))
        mods = Counter(module_of(f, 2) for f, _ in entries)
        story.append(Paragraph(
            "Concentrated in: " + ", ".join(f"<b>{m}</b> ({n})"
                                            for m, n in mods.most_common(4)), st["body"]))
        rows = [[Paragraph("File", st["cellh"]), Paragraph("Alignment", st["cellh"])]]
        for f, s in entries:
            rows.append([Paragraph(short(f, 3), st["cell"]),
                         Paragraph(f"{s:.3f}", st["cell"])])
        t = Table(rows, colWidths=[5.4 * inch, 0.9 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f6f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
    return story


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate Groundwork knowledge base reports as PDFs")
    ap.add_argument("--repo", required=True, help="Repository name")
    ap.add_argument("--type", required=True,
                    choices=["overview", "file", "capability"],
                    help="overview = repo summary + key point sections; "
                         "file = one file in context; "
                         "capability = capability flow diagram")
    ap.add_argument("--file", default=None, help="Target file (--type file)")
    ap.add_argument("--repo-path", default=None, help="Source location on disk")
    ap.add_argument("--out", default=None, help="Output PDF path")
    ap.add_argument("--reports-dir", default="reports",
                    help="Folder for reports (default: reports)")
    ap.add_argument("--files-per-capability", type=int, default=MAX_FILES_PER_CAPABILITY,
                    help=f"Files shown per capability (default: {MAX_FILES_PER_CAPABILITY})")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the LLM purpose inference in the overview")
    ap.add_argument("--keep-images", action="store_true",
                    help="Keep the generated PNGs next to the PDF")
    args = ap.parse_args()

    repo = args.repo
    repo_path = None
    if args.repo_path:
        repo_path = Path(args.repo_path)
    else:
        for cand in (Path("./repos") / repo, Path(repo)):
            if cand.is_dir():
                repo_path = cand
                break

    print(f"\n  Repository : {repo}")
    print(f"  Report     : {args.type}")

    print("  Loading knowledge base...")
    files      = fetch_files(repo)
    key_points = fetch_key_points(repo)
    vectors    = fetch_vectors(repo)
    edges      = fetch_edges(repo)
    print(f"    {len(files)} files, {len(key_points)} key points, "
          f"{len(vectors)} vectors, {len(edges)} edges")

    default_names = {
        "overview":   f"{repo}_overview.pdf",
        "file":       f"{repo}_{Path(args.file).stem}_file_report.pdf" if args.file else "file_report.pdf",
        "capability": f"{repo}_capability_flow.pdf",
    }
    out_path = Path(args.out) if args.out else Path(args.reports_dir) / default_names[args.type]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgdir = (out_path.parent / f"{repo}_{args.type}_figures") if args.keep_images \
             else Path(tempfile.mkdtemp(prefix="gw_rep_"))
    imgdir.mkdir(parents=True, exist_ok=True)

    print("  Building report...")
    if args.type == "overview":
        story = build_overview(repo, files, key_points, vectors, edges,
                               imgdir, use_llm=not args.no_llm)
        title = f"Repository Overview — {repo}"
    elif args.type == "file":
        if not args.file:
            sys.exit("--type file requires --file <path>")
        story = build_file_report(repo, args.file, repo_path, files, key_points,
                                  vectors, edges, imgdir)
        title = f"File Report — {args.file}"
    else:
        story = build_capability_report(repo, key_points, vectors, files, imgdir,
                                        args.files_per_capability)
        title = f"Capability System Diagram — {repo}"

    _doc(out_path, title).build(story)
    print(f"\n  ✓ Wrote {out_path}")
    if args.keep_images:
        print(f"  ✓ Figures in {imgdir}")
    print()


if __name__ == "__main__":
    main()