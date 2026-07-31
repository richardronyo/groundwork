#!/usr/bin/env python3
"""
Groundwork TUI — one terminal app for the whole pipeline.

Four tabs:
  Ingest    Run ingestion_pipeline.sh (all stages, one stage, or resume from
            a stage), with live streaming output.
  Diagrams  Generate any kb.diagram type (deps / function / class / keypoint
            / system) as Mermaid text and/or a PNG.
  Reports   Generate a full system_report.py PDF.
  Browse    Read-only viewer over PostgreSQL (files / business_rules /
            key_points), Kùzu (File nodes + DEPENDS_ON), and ChromaDB
            (per-repo vector collections) — with search/filter.

Run from the PROJECT ROOT (the directory containing kb/):
    python3 groundwork_tui.py

Requirements:
    pip install textual psycopg[binary] kuzu chromadb python-dotenv
(psycopg / kuzu / chromadb are only needed for the Browse tab — Ingest,
Diagrams, and Reports just shell out to the existing scripts, which already
depend on them.)
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, TabbedContent, TabPane, Input, Select, Button,
    RichLog, DataTable, Checkbox, Label, Static,
)
from textual.worker import Worker

PROJECT_ROOT = Path.cwd()
STAGES = ["init", "scan", "deps", "rules", "synth", "embed"]
DIAGRAM_TYPES = ["deps", "class"]
FORMATS = ["mermaid", "png", "both"]

WRITE_RE = re.compile(r"✓ Wrote (.+\.(?:png|jpe?g|pdf|md))\s*$")


def open_in_os(path: str) -> None:
    """Best-effort 'reveal this file' — macOS `open`, falling back quietly elsewhere."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", path])
        else:
            subprocess.Popen(["start", "", path], shell=True)
    except Exception:
        pass


async def stream_subprocess(cmd: list[str], log: RichLog, cwd: Path = PROJECT_ROOT,
                            stdin_data: bytes = None) -> tuple[int, str]:
    """
    Runs cmd, streaming stdout+stderr into `log` as it arrives.
    Returns (returncode, full_output) — full_output is used to scrape
    "✓ Wrote <path>" lines for the Open-file buttons.

    Reads raw bytes and splits on \\r OR \\n, rather than using
    StreamReader.readline() (which waits specifically for \\n). The pipeline's
    progress bars (json_to_graph.py, metadata.py, etc.) redraw in place with
    \\r and don't emit a real \\n until they're done — for a big repo those
    \\r-only updates can pile up past asyncio's internal ~64KB readline()
    buffer before a \\n ever shows up, and readline() raises
    "Separator is not found, and chunk exceed the limit" rather than just
    handling it. Splitting manually has no such limit.
    """
    log.write(f"[bold cyan]$ {' '.join(cmd)}[/bold cyan]")
    output_lines = []
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
        )
    except FileNotFoundError as e:
        log.write(f"[bold red]Failed to start: {e}[/bold red]")
        return 1, ""

    if stdin_data is not None:
        proc.stdin.write(stdin_data)
        await proc.stdin.drain()
        proc.stdin.close()

    def emit(raw: bytes, is_progress: bool) -> None:
        nonlocal progress_counter
        text = raw.decode(errors="replace").rstrip()
        if not text:
            return
        output_lines.append(text)
        if is_progress:
            # \r-terminated segments are progress-bar redraws (json_to_graph.py,
            # metadata.py, etc. all print "\r  [bar] i/total" in a loop with no
            # \n until they're done). Logging every single one floods the pane
            # with thousands of near-identical lines on a big repo — RichLog has
            # no "replace the last line" API, so throttle instead: only show
            # every 25th update. Every real (\n-terminated) line always shows.
            progress_counter += 1
            if progress_counter % 25 != 0:
                return
        log.write(text)

    progress_counter = 0
    buf = b""
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            idx_n = buf.find(b"\n")
            idx_r = buf.find(b"\r")
            candidates = [i for i in (idx_n, idx_r) if i != -1]
            if not candidates:
                break
            idx = min(candidates)
            emit(buf[:idx], is_progress=(buf[idx:idx + 1] == b"\r"))
            buf = buf[idx + 1:]
    if buf:
        emit(buf, is_progress=False)

    await proc.wait()
    status = "[bold green]✓ done[/bold green]" if proc.returncode == 0 \
        else f"[bold red]✗ exited {proc.returncode}[/bold red]"
    log.write(status)
    return proc.returncode, "\n".join(output_lines)


def find_written_file(output: str, *exts: str) -> str | None:
    """Scrapes the last '✓ Wrote <path>' line matching any of exts from captured output."""
    match = None
    for line in output.splitlines():
        m = WRITE_RE.search(line)
        if m and (not exts or Path(m.group(1)).suffix.lstrip(".").lower() in exts):
            match = m.group(1)
    return match


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestPane(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Repository path or GitHub URL")
        yield Input(placeholder="./repos/flask  or  https://github.com/user/repo", id="repo_input")
        with Horizontal(classes="row"):
            yield Label("Run:", classes="field-label")
            yield Select(
                [("All stages", "all"), ("Only one stage", "only"), ("From a stage onward", "from")],
                value="all", id="mode_select", allow_blank=False,
            )
            yield Label("Stage:", classes="field-label")
            yield Select([(s, s) for s in STAGES], value="init", id="stage_select", allow_blank=False)
        with Horizontal(classes="row"):
            yield Label("Workers", classes="field-label")
            yield Input(value="5", id="workers_input")
            yield Label("Embed workers", classes="field-label")
            yield Input(value="4", id="embed_workers_input")
            yield Label("Rate limit", classes="field-label")
            yield Input(value="60", id="rate_limit_input")
        yield Checkbox("Resume rules (only files not yet processed)", id="resume_checkbox")
        yield Button("Run ingestion", id="run_button", variant="primary")
        yield RichLog(id="ingest_log", wrap=True, markup=True, highlight=False, max_lines=5000)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run_button":
            self.run_worker(self.run_pipeline(), exclusive=True, group="ingest")

    async def run_pipeline(self) -> None:
        log = self.query_one("#ingest_log", RichLog)
        log.clear()

        repo = self.query_one("#repo_input", Input).value.strip()
        if not repo:
            log.write("[bold red]Enter a repo path or GitHub URL first.[/bold red]")
            return

        script = PROJECT_ROOT / "ingestion_pipeline.sh"
        if not script.exists():
            log.write(f"[bold red]Not found: {script}[/bold red]")
            return

        cmd = [str(script), repo]
        mode = self.query_one("#mode_select", Select).value
        stage = self.query_one("#stage_select", Select).value
        if mode == "only":
            cmd += ["--only", stage]
        elif mode == "from":
            cmd += ["--from", stage]

        workers = self.query_one("#workers_input", Input).value.strip()
        if workers:
            cmd += ["--workers", workers]
        embed_workers = self.query_one("#embed_workers_input", Input).value.strip()
        if embed_workers:
            cmd += ["--embed-workers", embed_workers]
        rate_limit = self.query_one("#rate_limit_input", Input).value.strip()
        if rate_limit:
            cmd += ["--rate-limit", rate_limit]
        if self.query_one("#resume_checkbox", Checkbox).value:
            cmd += ["--resume-rules"]

        # The script's own "Proceed? [Y/n]" prompt — answer it immediately.
        await stream_subprocess(cmd, log, stdin_data=b"Y\n")


# ── Diagrams ────────────────────────────────────────────────────────────────

class DiagramsPane(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Label("Repo", classes="field-label")
            yield Input(placeholder="flask", id="d_repo")
        with Horizontal(classes="row"):
            yield Label("Type", classes="field-label")
            yield Select([(t, t) for t in DIAGRAM_TYPES], value="deps", id="d_type", allow_blank=False)
            yield Label("Format", classes="field-label")
            yield Select([(f, f) for f in FORMATS], value="mermaid", id="d_format", allow_blank=False)
        with Horizontal(classes="row"):
            yield Label("File", classes="field-label")
            yield Input(placeholder="flask/src/flask/app.py — required for class, "
                                    "optional anchor for deps", id="d_file")
        with Horizontal(classes="row"):
            yield Label("Depth", classes="field-label")
            yield Input(placeholder="deps: hops from --file (default 1)", id="d_depth")
            yield Label("Max nodes", classes="field-label")
            yield Input(placeholder="deps: whole-repo view cap (default 60)", id="d_max_nodes")
            yield Label("Repo path", classes="field-label")
            yield Input(placeholder="class: optional — auto-detected under ./repos", id="d_repo_path")
        with Horizontal(classes="row"):
            yield Button("Generate", id="d_generate", variant="primary")
            yield Button("Open image", id="d_open", disabled=True)
        yield RichLog(id="d_log", wrap=True, markup=True, highlight=False, max_lines=5000)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "d_generate":
            self.run_worker(self.generate(), exclusive=True, group="diagrams")
        elif event.button.id == "d_open":
            path = getattr(self, "_last_image", None)
            if path:
                open_in_os(path)

    async def generate(self) -> None:
        log = self.query_one("#d_log", RichLog)
        log.clear()
        self.query_one("#d_open", Button).disabled = True

        repo = self.query_one("#d_repo", Input).value.strip()
        if not repo:
            log.write("[bold red]Enter a repo name first.[/bold red]")
            return

        dtype = self.query_one("#d_type", Select).value
        file_val = self.query_one("#d_file", Input).value.strip()
        if dtype == "class" and not file_val:
            log.write("[bold red]--type class requires a File (the file to diagram).[/bold red]")
            return

        fmt = self.query_one("#d_format", Select).value
        cmd = [sys.executable, "-m", "kb.diagram", "--repo", repo, "--type", dtype, "--format", fmt]

        def add(flag: str, widget_id: str) -> None:
            val = self.query_one(f"#{widget_id}", Input).value.strip()
            if val:
                cmd.extend([flag, val])

        add("--file", "d_file")
        add("--depth", "d_depth")
        add("--max-nodes", "d_max_nodes")
        add("--repo-path", "d_repo_path")

        rc, output = await stream_subprocess(cmd, log)
        if rc == 0:
            img = find_written_file(output, "png", "jpg", "jpeg")
            if img:
                self._last_image = img
                self.query_one("#d_open", Button).disabled = False


# ── Reports ─────────────────────────────────────────────────────────────────

class ReportsPane(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Label("Repo", classes="field-label")
            yield Input(placeholder="nopCommerce", id="r_repo")
            yield Label("Repo path", classes="field-label")
            yield Input(placeholder="optional — auto-detected under ./repos", id="r_repo_path")
        with Horizontal(classes="row"):
            yield Label("Module depth", classes="field-label")
            yield Input(placeholder="2", id="r_module_depth")
            yield Label("Max nodes", classes="field-label")
            yield Input(placeholder="30", id="r_max_nodes")
            yield Label("Max funcs", classes="field-label")
            yield Input(placeholder="14", id="r_max_funcs")
        with Horizontal(classes="row"):
            yield Label("Scan files", classes="field-label")
            yield Input(placeholder="600", id="r_scan_files")
            yield Label("Appendix items", classes="field-label")
            yield Input(placeholder="6", id="r_appendix_items")
            yield Checkbox("Keep images", id="r_keep_images")
        with Horizontal(classes="row"):
            yield Label("Output PDF", classes="field-label")
            yield Input(placeholder="optional — default reports/<repo>_system_report.pdf", id="r_out")
        with Horizontal(classes="row"):
            yield Button("Generate report", id="r_generate", variant="primary")
            yield Button("Open PDF", id="r_open", disabled=True)
        yield RichLog(id="r_log", wrap=True, markup=True, highlight=False, max_lines=5000)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "r_generate":
            self.run_worker(self.generate(), exclusive=True, group="reports")
        elif event.button.id == "r_open":
            path = getattr(self, "_last_pdf", None)
            if path:
                open_in_os(path)

    async def generate(self) -> None:
        log = self.query_one("#r_log", RichLog)
        log.clear()
        self.query_one("#r_open", Button).disabled = True

        repo = self.query_one("#r_repo", Input).value.strip()
        if not repo:
            log.write("[bold red]Enter a repo name first.[/bold red]")
            return

        cmd = [sys.executable, "-m", "system_report", "--repo", repo]

        def add(flag: str, widget_id: str) -> None:
            val = self.query_one(f"#{widget_id}", Input).value.strip()
            if val:
                cmd.extend([flag, val])

        add("--repo-path", "r_repo_path")
        add("--module-depth", "r_module_depth")
        add("--max-nodes", "r_max_nodes")
        add("--max-funcs", "r_max_funcs")
        add("--scan-files", "r_scan_files")
        add("--appendix-items", "r_appendix_items")
        add("--out", "r_out")
        if self.query_one("#r_keep_images", Checkbox).value:
            cmd.append("--keep-images")

        rc, output = await stream_subprocess(cmd, log)
        if rc == 0:
            pdf = find_written_file(output, "pdf")
            if pdf:
                self._last_pdf = pdf
                self.query_one("#r_open", Button).disabled = False



# ── DB helpers (Browse tab) — each degrades gracefully if a store is unreachable ──

def pg_list_repos() -> tuple[list[str], str | None]:
    try:
        from kb.relationaldb.initialize_db import get_connection, list_repositories
        conn = get_connection()
        try:
            return list_repositories(conn), None
        finally:
            conn.close()
    except Exception as e:
        return [], str(e)


def pg_files(repo: str) -> tuple[list[dict], str | None]:
    try:
        from kb.relationaldb.initialize_db import get_connection
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT file_path, language, lines, classes, functions, methods, rules_extracted
                    FROM files WHERE repository_name = %s ORDER BY file_path
                """, (repo,))
                rows = cur.fetchall()
            return [
                {"file_path": r[0], "language": r[1], "lines": r[2], "classes": r[3],
                 "functions": r[4], "methods": r[5], "rules_extracted": r[6]}
                for r in rows
            ], None
        finally:
            conn.close()
    except Exception as e:
        return [], str(e)


def pg_rules(repo: str) -> tuple[dict, str | None]:
    try:
        from kb.relationaldb.initialize_db import get_connection, load_business_rules_from_db
        conn = get_connection()
        try:
            return load_business_rules_from_db(conn, repo), None
        finally:
            conn.close()
    except Exception as e:
        return {}, str(e)


def pg_key_points(repo: str) -> tuple[list[str], str | None]:
    try:
        from kb.relationaldb.initialize_db import get_connection, load_key_points_from_db
        conn = get_connection()
        try:
            return load_key_points_from_db(conn, repo), None
        finally:
            conn.close()
    except Exception as e:
        return [], str(e)


def kuzu_list_repos() -> tuple[list[str], str | None]:
    try:
        from kb.graph.kuzu_store import get_connection, rows_to_dicts
        res = get_connection().execute("MATCH (f:File) RETURN DISTINCT f.repository_name")
        return sorted(r[0] for r in rows_to_dicts(res)), None
    except Exception as e:
        return [], str(e)


def kuzu_files(repo: str) -> tuple[list[dict], str | None]:
    try:
        from kb.graph.kuzu_store import get_connection, rows_to_dicts
        res = get_connection().execute("""
            MATCH (f:File {repository_name: $repo})
            RETURN f.relative, f.language, f.size_bytes
            ORDER BY f.relative
        """, {"repo": repo})
        rows = rows_to_dicts(res)
        return [{"relative": r[0], "language": r[1], "size_bytes": r[2]} for r in rows], None
    except Exception as e:
        return [], str(e)


def kuzu_deps(repo: str, relative: str) -> tuple[list[str], list[str], str | None]:
    try:
        from kb.graph.kuzu_store import get_connection, rows_to_dicts
        conn = get_connection()
        out_res = conn.execute("""
            MATCH (a:File {repository_name: $repo, relative: $rel})-[:DEPENDS_ON]->(b:File)
            RETURN b.relative
        """, {"repo": repo, "rel": relative})
        in_res = conn.execute("""
            MATCH (a:File)-[:DEPENDS_ON]->(b:File {repository_name: $repo, relative: $rel})
            RETURN a.relative
        """, {"repo": repo, "rel": relative})
        outgoing = [r[0] for r in rows_to_dicts(out_res)]
        incoming = [r[0] for r in rows_to_dicts(in_res)]
        return outgoing, incoming, None
    except Exception as e:
        return [], [], str(e)


def chroma_list_collections() -> tuple[list[str], str | None]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        return sorted(c.name for c in client.list_collections()), None
    except Exception as e:
        return [], str(e)


def chroma_entries(collection: str) -> tuple[list[dict], str | None]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_collection(collection)
        got = col.get(include=["metadatas"])
        entries = []
        for i, meta in zip(got["ids"], got["metadatas"]):
            meta = meta or {}
            entries.append({
                "id": i,
                "relative": meta.get("relative", ""),
                "name": meta.get("name", ""),
                "rule_count": meta.get("rule_count", ""),
                "_meta": meta,
            })
        entries.sort(key=lambda e: e["relative"] or e["id"])
        return entries, None
    except Exception as e:
        return [], str(e)


# ── Browse: PostgreSQL ────────────────────────────────────────────────────────

class PostgresBrowser(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Select([], id="pg_repo", allow_blank=True, prompt="Select repo...")
            yield Button("Refresh repos", id="pg_refresh")
            yield Input(placeholder="Filter by path...", id="pg_filter")
        yield DataTable(id="pg_table", zebra_stripes=True, cursor_type="row")
        yield Static("Select a file to see its business rules and the repo's key points.",
                     id="pg_detail")

    def on_mount(self) -> None:
        table = self.query_one("#pg_table", DataTable)
        table.add_columns("File", "Language", "Lines", "Classes", "Functions", "Methods", "Rules?")
        self._all_files: list[dict] = []
        self._rules: dict = {}
        self._key_points: list[str] = []
        self.refresh_repos()

    def refresh_repos(self) -> None:
        repos, err = pg_list_repos()
        self.query_one("#pg_repo", Select).set_options([(r, r) for r in repos])
        if err:
            self.query_one("#pg_detail", Static).update(f"[red]PostgreSQL unavailable: {err}[/red]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pg_refresh":
            self.refresh_repos()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "pg_repo" and event.value != Select.BLANK:
            self.load_repo(str(event.value))

    def load_repo(self, repo: str) -> None:
        detail = self.query_one("#pg_detail", Static)
        files, err = pg_files(repo)
        self._rules, _ = pg_rules(repo)
        self._key_points, _ = pg_key_points(repo)
        self._all_files = files
        self.render_table(files)
        if err:
            detail.update(f"[red]{err}[/red]")
            return
        kp_text = "\n".join(f"  {i + 1}. {kp[:120]}" for i, kp in enumerate(self._key_points[:5]))
        total_rules = sum(len(v) for v in self._rules.values())
        detail.update(f"[bold]{repo}[/bold] — {len(files)} files, {total_rules} rules, "
                      f"{len(self._key_points)} key points\n\n"
                      f"[bold]Key points (first 5):[/bold]\n{kp_text or '  (none)'}")

    def render_table(self, files: list[dict]) -> None:
        table = self.query_one("#pg_table", DataTable)
        table.clear()
        for f in files:
            table.add_row(
                f["file_path"], f["language"] or "", str(f["lines"] or 0),
                str(f["classes"] or 0), str(f["functions"] or 0), str(f["methods"] or 0),
                "✓" if f["rules_extracted"] else "",
                key=f["file_path"],
            )

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "pg_filter":
            needle = event.value.lower()
            self.render_table([f for f in self._all_files if needle in f["file_path"].lower()])

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "pg_table":
            return
        path = str(event.row_key.value)
        rules = self._rules.get(path, [])
        body = "\n".join(f"  • {r}" for r in rules) if rules else "  (no business rules extracted)"
        self.query_one("#pg_detail", Static).update(
            f"[bold]{path}[/bold]\n\n[bold]Business rules:[/bold]\n{body}")


# ── Browse: Kùzu ──────────────────────────────────────────────────────────────

class KuzuBrowser(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Select([], id="kz_repo", allow_blank=True, prompt="Select repo...")
            yield Button("Refresh repos", id="kz_refresh")
            yield Input(placeholder="Filter by path...", id="kz_filter")
        yield DataTable(id="kz_table", zebra_stripes=True, cursor_type="row")
        yield Static("Select a file to see its DEPENDS_ON edges.", id="kz_detail")

    def on_mount(self) -> None:
        table = self.query_one("#kz_table", DataTable)
        table.add_columns("File", "Language", "Size (bytes)")
        self._all_files: list[dict] = []
        self._repo = ""
        self.refresh_repos()

    def refresh_repos(self) -> None:
        repos, err = kuzu_list_repos()
        self.query_one("#kz_repo", Select).set_options([(r, r) for r in repos])
        if err:
            self.query_one("#kz_detail", Static).update(f"[red]Kùzu unavailable: {err}[/red]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "kz_refresh":
            self.refresh_repos()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "kz_repo" and event.value != Select.BLANK:
            self.load_repo(str(event.value))

    def load_repo(self, repo: str) -> None:
        self._repo = repo
        files, err = kuzu_files(repo)
        self._all_files = files
        self.render_table(files)
        detail = self.query_one("#kz_detail", Static)
        detail.update(f"[red]{err}[/red]" if err else f"[bold]{repo}[/bold] — {len(files)} File nodes")

    def render_table(self, files: list[dict]) -> None:
        table = self.query_one("#kz_table", DataTable)
        table.clear()
        for f in files:
            table.add_row(f["relative"], f["language"] or "", str(f["size_bytes"] or 0), key=f["relative"])

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "kz_filter":
            needle = event.value.lower()
            self.render_table([f for f in self._all_files if needle in f["relative"].lower()])

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "kz_table":
            return
        rel = str(event.row_key.value)
        outgoing, incoming, err = kuzu_deps(self._repo, rel)
        detail = self.query_one("#kz_detail", Static)
        if err:
            detail.update(f"[red]{err}[/red]")
            return
        out_txt = "\n".join(f"  → {o}" for o in outgoing) or "  (none)"
        in_txt = "\n".join(f"  ← {i}" for i in incoming) or "  (none)"
        detail.update(f"[bold]{rel}[/bold]\n\n"
                      f"[bold]Depends on ({len(outgoing)}):[/bold]\n{out_txt}\n\n"
                      f"[bold]Depended on by ({len(incoming)}):[/bold]\n{in_txt}")


# ── Browse: ChromaDB ────────────────────────────────────────────────────────

class ChromaBrowser(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Select([], id="ch_repo", allow_blank=True, prompt="Select collection...")
            yield Button("Refresh collections", id="ch_refresh")
            yield Input(placeholder="Filter by path...", id="ch_filter")
        yield DataTable(id="ch_table", zebra_stripes=True, cursor_type="row")
        yield Static("Select an entry to see its full metadata.", id="ch_detail")

    def on_mount(self) -> None:
        table = self.query_one("#ch_table", DataTable)
        table.add_columns("ID", "Relative", "Rule count")
        self._all_entries: list[dict] = []
        self._by_id: dict = {}
        self.refresh_collections()

    def refresh_collections(self) -> None:
        cols, err = chroma_list_collections()
        self.query_one("#ch_repo", Select).set_options([(c, c) for c in cols])
        if err:
            self.query_one("#ch_detail", Static).update(f"[red]ChromaDB unavailable: {err}[/red]")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ch_refresh":
            self.refresh_collections()

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "ch_repo" and event.value != Select.BLANK:
            self.load_collection(str(event.value))

    def load_collection(self, name: str) -> None:
        entries, err = chroma_entries(name)
        self._all_entries = entries
        self._by_id = {e["id"]: e for e in entries}
        self.render_table(entries)
        detail = self.query_one("#ch_detail", Static)
        detail.update(f"[red]{err}[/red]" if err else f"[bold]{name}[/bold] — {len(entries)} entries")

    def render_table(self, entries: list[dict]) -> None:
        table = self.query_one("#ch_table", DataTable)
        table.clear()
        for e in entries:
            table.add_row(e["id"][:60], e["relative"], str(e["rule_count"]), key=e["id"])

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "ch_filter":
            needle = event.value.lower()
            self.render_table([e for e in self._all_entries
                               if needle in (e["relative"] or "").lower() or needle in e["id"].lower()])

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "ch_table":
            return
        entry = self._by_id.get(str(event.row_key.value))
        if not entry:
            return
        meta_lines = "\n".join(f"  {k}: {v}" for k, v in entry["_meta"].items())
        self.query_one("#ch_detail", Static).update(
            f"[bold]{entry['id']}[/bold]\n\n[bold]Metadata:[/bold]\n{meta_lines}")


class TestsPane(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Label("Repo", classes="field-label")
            yield Input(placeholder="optional if only one repo in the DB", id="t_repo")
            yield Label("Repo path", classes="field-label")
            yield Input(placeholder="optional — auto-detected under ./repos", id="t_repo_path")
        with Horizontal(classes="row"):
            yield Label("File", classes="field-label")
            yield Input(placeholder="src/flask/app.py — or leave blank and use Prompt", id="t_file")
            yield Label("Prompt", classes="field-label")
            yield Input(placeholder="finds the file most associated with this — overrides File",
                        id="t_prompt")
        with Horizontal(classes="row"):
            yield Label("Example test file", classes="field-label")
            yield Input(placeholder="optional — path to an existing test file to match style/"
                                    "framework against, overriding auto-detection", id="t_example_file")
        with Horizontal(classes="row"):
            yield Label("Function", classes="field-label")
            yield Input(placeholder="optional — a single function/method to target", id="t_function")
            yield Label("Related", classes="field-label")
            yield Input(placeholder="3", id="t_related")
            yield Label("Min score", classes="field-label")
            yield Input(placeholder="0.3 — only used with Prompt", id="t_min_score")
        with Horizontal(classes="row"):
            yield Label("Tests dir", classes="field-label")
            yield Input(placeholder="generated_tests", id="t_out")
            yield Label("Reports dir", classes="field-label")
            yield Input(placeholder="weakness", id="t_reports_dir")
            yield Label("Report format", classes="field-label")
            yield Select([("pdf", "pdf"), ("md", "md"), ("both", "both")],
                        value="pdf", id="t_report_format", allow_blank=False)
        with Horizontal(classes="row"):
            yield Checkbox("Follow imports (also test imported functions)", id="t_follow_imports")
            yield Checkbox("Skip weakness report", id="t_no_analyze")
            yield Checkbox("Print to log instead of writing files", id="t_print")
        with Horizontal(classes="row"):
            yield Button("Generate tests", id="t_generate", variant="primary")
            yield Button("Open tests folder", id="t_open_tests", disabled=True)
            yield Button("Open weakness folder", id="t_open_weakness", disabled=True)
        yield RichLog(id="t_log", wrap=True, markup=True, highlight=False, max_lines=5000)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "t_generate":
            self.run_worker(self.generate(), exclusive=True, group="tests")
        elif event.button.id == "t_open_tests":
            path = getattr(self, "_tests_dir", None)
            if path:
                open_in_os(path)
        elif event.button.id == "t_open_weakness":
            path = getattr(self, "_weakness_dir", None)
            if path:
                open_in_os(path)

    async def generate(self) -> None:
        log = self.query_one("#t_log", RichLog)
        log.clear()
        self.query_one("#t_open_tests", Button).disabled = True
        self.query_one("#t_open_weakness", Button).disabled = True

        file_val = self.query_one("#t_file", Input).value.strip()
        prompt_val = self.query_one("#t_prompt", Input).value.strip()
        if not file_val and not prompt_val:
            log.write("[bold red]Enter a File or a Prompt — generate_tests.py needs one of them.[/bold red]")
            return

        cmd = [sys.executable, "-m", "generate_tests"]

        def add(flag: str, widget_id: str) -> None:
            val = self.query_one(f"#{widget_id}", Input).value.strip()
            if val:
                cmd.extend([flag, val])

        add("--repo", "t_repo")
        # --prompt takes precedence over --file, matching generate_tests.py's
        # own resolution order (it checks args.prompt before args.file).
        if prompt_val:
            cmd.extend(["--prompt", prompt_val])
        else:
            cmd.extend(["--file", file_val])
        add("--function", "t_function")
        add("--example-file", "t_example_file")
        add("--repo-path", "t_repo_path")
        add("--related", "t_related")
        add("--min-score", "t_min_score")

        tests_dir = self.query_one("#t_out", Input).value.strip() or "generated_tests"
        cmd.extend(["--out", tests_dir])
        weakness_dir = self.query_one("#t_reports_dir", Input).value.strip() or "weakness"
        cmd.extend(["--reports-dir", weakness_dir])
        cmd.extend(["--report-format", self.query_one("#t_report_format", Select).value])

        follow_imports = self.query_one("#t_follow_imports", Checkbox).value
        no_analyze = self.query_one("#t_no_analyze", Checkbox).value
        to_stdout = self.query_one("#t_print", Checkbox).value
        if follow_imports:
            cmd.append("--follow-imports")
        if no_analyze:
            cmd.append("--no-analyze")
        if to_stdout:
            cmd.append("--print")

        rc, _ = await stream_subprocess(cmd, log)
        if rc == 0 and not to_stdout:
            self._tests_dir = tests_dir
            self._weakness_dir = weakness_dir
            self.query_one("#t_open_tests", Button).disabled = False
            if not no_analyze:
                self.query_one("#t_open_weakness", Button).disabled = False


class BrowsePane(Vertical):
    def compose(self) -> ComposeResult:
        with TabbedContent(initial="pg"):
            with TabPane("PostgreSQL", id="pg"):
                yield PostgresBrowser()
            with TabPane("Kùzu", id="kz"):
                yield KuzuBrowser()
            with TabPane("ChromaDB", id="ch"):
                yield ChromaBrowser()


# ── App ─────────────────────────────────────────────────────────────────────

class GroundworkApp(App):
    TITLE = "Groundwork"
    SUB_TITLE = str(PROJECT_ROOT)

    CSS = """
    .row { height: auto; align: left middle; margin-bottom: 1; overflow-x: auto; }
    .field-label { width: auto; margin: 0 1; content-align: right middle; color: $text-muted; }
    Input { width: 1fr; margin-right: 1; }
    Select { width: 22; margin-right: 1; }
    RichLog { height: 1fr; border: round $primary; margin-top: 1; }
    DataTable { height: 14; border: round $primary; }
    #pg_detail, #kz_detail, #ch_detail { height: 1fr; border: round $primary; padding: 1; overflow-y: auto; }
    #root_warning { background: $warning; color: $text; padding: 0 1; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        if not (PROJECT_ROOT / "kb").is_dir():
            yield Static(
                f"⚠  No 'kb' directory found in {PROJECT_ROOT} — run this from the project root.",
                id="root_warning",
            )
        with TabbedContent(initial="ingest"):
            with TabPane("Ingest", id="ingest"):
                yield IngestPane()
            with TabPane("Diagrams", id="diagrams"):
                yield DiagramsPane()
            with TabPane("Reports", id="reports"):
                yield ReportsPane()
            with TabPane("Tests", id="tests"):
                yield TestsPane()
            with TabPane("Browse", id="browse"):
                yield BrowsePane()
        yield Footer()


def main() -> None:
    GroundworkApp().run()


if __name__ == "__main__":
    main()