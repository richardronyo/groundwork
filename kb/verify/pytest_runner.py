"""
Groundwork — run a generated pytest file against a repository and report.

Used by the agentic test-generation graph's `run_tests` node. The test file is
written to a fresh temp directory (never into the repo), and the repo root,
its `src/` folder and the folder containing the repo are put on PYTHONPATH so
both `import flask` (src layout) and `import pricing` (flat layout) resolve.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_OUTPUT_CHARS = 4000
DEFAULT_TIMEOUT = 120

_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed|skipped)")


def _pythonpath(repo_root: Path) -> str:
    parts = [str(repo_root), str(repo_root / "src"), str(repo_root.parent)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(p for p in parts if p)


def run_pytest(test_code: str, repo_root: Path | str, filename: str = "test_generated.py",
               timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Returns {
        "passed": bool,          # pytest exit code 0 (xfail counts as passing)
        "counts": {"passed": n, "failed": n, "error": n, "xfailed": n, ...},
        "exit_code": int,
        "output": str,           # tail of pytest output, for the reflect node
    }
    """
    repo_root = Path(repo_root).resolve()
    with tempfile.TemporaryDirectory(prefix="gw_pytest_") as tmp:
        path = Path(tmp) / filename
        path.write_text(test_code)
        env = {**os.environ, "PYTHONPATH": _pythonpath(repo_root), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
                 str(path)],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
            output, code = proc.stdout + proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            output = f"TIMEOUT after {timeout}s\n{(e.stdout or '')}"
            code = -1

    counts: dict[str, int] = {}
    for n, kind in _COUNT_RE.findall(output):
        counts["error" if kind.startswith("error") else kind] = int(n)
    # exit 5 = "no tests collected": not a pass even though nothing failed.
    return {"passed": code == 0, "counts": counts, "exit_code": code,
            "output": output[-MAX_OUTPUT_CHARS:]}
