#!/usr/bin/env python3
"""
Groundwork — Shared PDF rendering

One place that knows how to turn Groundwork's text output into PDFs, so
generate_tests.py and run_tests.py don't each reimplement ReportLab styling.

  markdown_to_pdf()  Renders the LLM's Markdown weaknesses report as a PDF.
  results_to_pdf()   Renders a parsed test run as a PDF.

Requirements:
    pip install reportlab
"""

import re
import html
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak, KeepTogether)

C_PRIMARY = "#2e5c8a"
C_ACCENT  = "#c1663a"
C_PASS    = "#2f7d4f"
C_FAIL    = "#b3352e"
C_XFAIL   = "#8a6d1f"
C_MUTED   = "#5a6570"


def _styles():
    s = getSampleStyleSheet()
    return {
        "title":  ParagraphStyle("t", parent=s["Title"], fontSize=21, spaceAfter=4),
        "sub":    ParagraphStyle("sub", parent=s["Normal"], fontSize=10.5,
                                 textColor=colors.HexColor(C_MUTED), spaceAfter=14),
        "h1":     ParagraphStyle("h1", parent=s["Heading1"], fontSize=15,
                                 textColor=colors.HexColor(C_PRIMARY),
                                 spaceBefore=14, spaceAfter=7),
        "h2":     ParagraphStyle("h2", parent=s["Heading2"], fontSize=12.5,
                                 textColor=colors.HexColor(C_ACCENT),
                                 spaceBefore=11, spaceAfter=5),
        "h3":     ParagraphStyle("h3", parent=s["Heading3"], fontSize=11,
                                 spaceBefore=9, spaceAfter=4),
        "body":   ParagraphStyle("b", parent=s["Normal"], fontSize=9.7,
                                 leading=14, spaceAfter=6),
        "bullet": ParagraphStyle("bu", parent=s["Normal"], fontSize=9.7, leading=14,
                                 leftIndent=14, bulletIndent=4, spaceAfter=3),
        "code":   ParagraphStyle("c", parent=s["Normal"], fontName="Courier",
                                 fontSize=7.6, leading=9.6,
                                 backColor=colors.HexColor("#f4f6f8"),
                                 textColor=colors.HexColor("#2b3138"),
                                 borderPadding=6, spaceBefore=4, spaceAfter=8,
                                 leftIndent=4),
        "cell":   ParagraphStyle("cl", parent=s["Normal"], fontSize=8.4, leading=11),
        "cellh":  ParagraphStyle("ch", parent=s["Normal"], fontSize=8.6, leading=11,
                                 textColor=colors.white, fontName="Helvetica-Bold"),
    }


def _inline(text: str) -> str:
    """Markdown inline formatting -> ReportLab markup, with everything escaped."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.6">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def _doc(out_path: Path, title: str):
    return SimpleDocTemplate(
        str(out_path), pagesize=letter,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch, title=title)


def markdown_to_pdf(md_text: str, out_path, title: str, subtitle: str = "") -> Path:
    """
    Renders Markdown (headings, bullets, numbered lists, fenced code, bold/italic
    /inline code) as a PDF. Built for the weaknesses reports the analysis pass
    produces, so anything it emits renders sensibly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st = _styles()
    story = [Paragraph(html.escape(title), st["title"])]
    if subtitle:
        story.append(Paragraph(html.escape(subtitle), st["sub"]))

    in_code, buf = False, []

    def flush_code():
        if buf:
            body = "<br/>".join(
                html.escape(l).replace(" ", "&nbsp;") for l in buf)
            story.append(Paragraph(body, st["code"]))
            buf.clear()

    for raw in md_text.split("\n"):
        line = raw.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            buf.append(line)
            continue

        if not line.strip():
            story.append(Spacer(1, 4))
            continue
        if set(line.strip()) <= {"-", "*", "_"} and len(line.strip()) >= 3:
            continue                                    # horizontal rule

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            key = "title" if level == 1 else ("h1" if level == 2 else
                  ("h2" if level == 3 else "h3"))
            story.append(Paragraph(_inline(m.group(2)), st[key]))
            continue

        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m:
            depth = len(m.group(1)) // 2
            sty = ParagraphStyle(f"bu{depth}", parent=st["bullet"],
                                 leftIndent=14 + 14 * depth,
                                 bulletIndent=4 + 14 * depth)
            story.append(Paragraph(_inline(m.group(2)), sty, bulletText="•"))
            continue

        m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)
        if m:
            depth = len(m.group(1)) // 2
            sty = ParagraphStyle(f"nu{depth}", parent=st["bullet"],
                                 leftIndent=18 + 14 * depth,
                                 bulletIndent=4 + 14 * depth)
            story.append(Paragraph(_inline(m.group(3)), sty,
                                   bulletText=f"{m.group(2)}."))
            continue

        story.append(Paragraph(_inline(line), st["body"]))

    flush_code()
    _doc(out_path, title).build(story)
    return out_path


# ── Test results ──────────────────────────────────────────────────────────────

STATUS_COLOR = {
    "PASSED":  C_PASS,  "FAILED": C_FAIL,  "ERROR": C_FAIL,
    "XFAIL":   C_XFAIL, "XPASS":  C_XFAIL, "SKIPPED": C_MUTED,
}


def results_to_pdf(run, out_path, title="Test Results", subtitle="") -> Path:
    """
    Renders a parsed test run.

    `run` is a dict:
      command   str
      target    str
      duration  str
      counts    {"passed": n, "failed": n, "xfailed": n, ...}
      tests     [{"name","status","file","message"}]
      raw_tail  str  (last lines of output, for context)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st = _styles()
    story = [Paragraph(html.escape(title), st["title"])]
    if subtitle:
        story.append(Paragraph(html.escape(subtitle), st["sub"]))

    counts = run.get("counts", {})
    total = sum(counts.values()) or len(run.get("tests", []))
    order = ["passed", "failed", "error", "xfailed", "xpassed", "skipped"]
    present = [(k, counts[k]) for k in order if counts.get(k)]

    rows = [[Paragraph("Outcome", st["cellh"]), Paragraph("Count", st["cellh"]),
             Paragraph("Meaning", st["cellh"])]]
    meaning = {
        "passed":  "Behaved as the test asserts.",
        "failed":  "Assertion failed or the test errored — needs attention.",
        "error":   "Could not run (import/collection error).",
        "xfailed": "Expected to fail: a documented weakness, still unfixed.",
        "xpassed": "Expected to fail but passed — the weakness may be fixed; "
                   "re-check the xfail marker.",
        "skipped": "Not run (untestable as written, or skipped deliberately).",
    }
    for k, n in present:
        rows.append([
            Paragraph(f'<font color="{STATUS_COLOR.get(k.upper(), C_MUTED)}">'
                      f"<b>{k}</b></font>", st["cell"]),
            Paragraph(str(n), st["cell"]),
            Paragraph(meaning.get(k, ""), st["cell"]),
        ])

    t = Table(rows, colWidths=[1.1 * inch, 0.7 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f6f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    meta_bits = []
    if run.get("target"):   meta_bits.append(f"<b>Target:</b> {html.escape(run['target'])}")
    if run.get("duration"): meta_bits.append(f"<b>Duration:</b> {html.escape(run['duration'])}")
    meta_bits.append(f"<b>Total:</b> {total}")
    story.append(Paragraph(" &nbsp;·&nbsp; ".join(meta_bits), st["body"]))
    if run.get("command"):
        story.append(Paragraph(
            f'<font face="Courier" size="8">{html.escape(run["command"])}</font>',
            st["body"]))

    # Failures first — that's what a reader needs
    tests = run.get("tests", [])
    bad = [t for t in tests if t["status"] in ("FAILED", "ERROR")]
    odd = [t for t in tests if t["status"] in ("XFAIL", "XPASS", "SKIPPED")]
    good = [t for t in tests if t["status"] == "PASSED"]

    if bad:
        story.append(Paragraph("Failures", st["h1"]))
        story.append(Paragraph(
            "Each failure below is either a real defect in the code under test or a "
            "problem with the generated test itself. Check the message before "
            "assuming which.", st["body"]))
        for t in bad:
            block = [Paragraph(
                f'<font color="{C_FAIL}"><b>{html.escape(t["name"])}</b></font>',
                st["h3"])]
            if t.get("message"):
                msg = t["message"][:900]
                block.append(Paragraph(
                    "<br/>".join(html.escape(l).replace(" ", "&nbsp;")
                                 for l in msg.split("\n")[:14]), st["code"]))
            story.append(KeepTogether(block))

    if odd:
        story.append(Paragraph("Expected failures, unexpected passes, and skips", st["h1"]))
        story.append(Paragraph(
            "These carry the weakness-probing tests. An <b>XFAIL</b> is a documented "
            "gap that still exists. An <b>XPASS</b> means the code now behaves "
            "correctly and the marker should be removed. A <b>SKIP</b> usually means "
            "the target could not be tested as written.", st["body"]))
        rows = [[Paragraph("Test", st["cellh"]), Paragraph("Status", st["cellh"]),
                 Paragraph("Reason", st["cellh"])]]
        for t in odd:
            rows.append([
                Paragraph(html.escape(t["name"]), st["cell"]),
                Paragraph(f'<font color="{STATUS_COLOR.get(t["status"], C_MUTED)}">'
                          f'<b>{t["status"]}</b></font>', st["cell"]),
                Paragraph(html.escape((t.get("message") or "")[:220]), st["cell"]),
            ])
        tt = Table(rows, colWidths=[2.7 * inch, 0.8 * inch, 2.8 * inch])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(C_PRIMARY)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6dbe0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f6f8")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tt)

    if good:
        story.append(Paragraph("Passing tests", st["h1"]))
        for t in good:
            story.append(Paragraph(
                f'<font color="{C_PASS}">✓</font> {html.escape(t["name"])}',
                st["bullet"]))

    if run.get("raw_tail"):
        story.append(PageBreak())
        story.append(Paragraph("Raw output (tail)", st["h1"]))
        story.append(Paragraph(
            "<br/>".join(html.escape(l).replace(" ", "&nbsp;")
                         for l in run["raw_tail"].split("\n")[-70:]), st["code"]))

    _doc(out_path, title).build(story)
    return out_path


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")