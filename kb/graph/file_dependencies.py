"""
Groundwork — Dependency Edge Builder
Reads the first N lines of each code file, extracts import statements,
resolves them against known files in the repo, and creates DEPENDS_ON
edges in the embedded Kùzu graph.

Usage:
    python3 build_dependencies.py <files.json> --repo <path/to/repo>
    python3 build_dependencies.py flask_files.json --repo ./flask --lines 30

Requirements:
    pip install kuzu python-dotenv
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from kb.graph.kuzu_store import (
    get_connection, uid_for, batched, scalar, KUZU_DB_PATH,
)

load_dotenv()

# ── Import extractors per language ────────────────────────────────────────────
# Each returns a list of raw import strings from a block of lines

def extract_python(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        # from x.y.z import ...
        m = re.match(r'^from\s+([\w.]+)\s+import', line)
        if m:
            imports.append(m.group(1))
            continue
        # import x.y.z
        m = re.match(r'^import\s+([\w.,\s]+)', line)
        if m:
            for part in m.group(1).split(","):
                imports.append(part.strip().split()[0])
    return imports

def extract_javascript(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        # import ... from './path'  or  import './path'
        m = re.search(r'''from\s+['"]([^'"]+)['"]''', line)
        if m:
            imports.append(m.group(1))
            continue
        # require('./path')
        m = re.search(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)''', line)
        if m:
            imports.append(m.group(1))
    return imports

def extract_csharp(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        # using Nop.Core.Domain.Customers;
        m = re.match(r'^using\s+([\w.]+)\s*;', line)
        if m:
            imports.append(m.group(1))
    return imports

def extract_java(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        m = re.match(r'^import\s+([\w.]+)\s*;', line)
        if m:
            imports.append(m.group(1))
    return imports

def extract_cpp(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        # #include "local.h" or #include <system.h>
        m = re.match(r'^#include\s+["<]([\w./]+)[">]', line)
        if m:
            imports.append(m.group(1))
    return imports

def extract_go(lines: list[str]) -> list[str]:
    imports = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == 'import (': in_block = True; continue
        if in_block and stripped == ')': in_block = False; continue
        if in_block:
            m = re.search(r'"([^"]+)"', stripped)
            if m: imports.append(m.group(1))
        else:
            m = re.match(r'^import\s+"([^"]+)"', stripped)
            if m: imports.append(m.group(1))
    return imports

def extract_ruby(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        m = re.match(r'^require(?:_relative)?\s+[\'"]([^\'"]+)[\'"]', line)
        if m:
            imports.append(m.group(1))
    return imports

def extract_rust(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        m = re.match(r'^use\s+([\w:]+)', line)
        if m:
            imports.append(m.group(1).replace("::", "."))
    return imports

def extract_shell(lines: list[str]) -> list[str]:
    imports = []
    for line in lines:
        line = line.strip()
        # source ./other.sh  or  . ./other.sh
        m = re.match(r'^(?:source|\.)\s+([\w./]+)', line)
        if m:
            imports.append(m.group(1))
    return imports

EXTRACTORS = {
    "Python":              extract_python,
    "JavaScript":          extract_javascript,
    "TypeScript":          extract_javascript,
    "JavaScript (React)":  extract_javascript,
    "TypeScript (React)":  extract_javascript,
    "C#":                  extract_csharp,
    "Java":                extract_java,
    "C++":                 extract_cpp,
    "C":                   extract_cpp,
    "C/C++ Header":        extract_cpp,
    "Go":                  extract_go,
    "Ruby":                extract_ruby,
    "Rust":                extract_rust,
    "Shell":               extract_shell,
    "Batch":               extract_shell,
}

# ── Import resolver ───────────────────────────────────────────────────────────

def build_lookup(files: list[dict]) -> dict:
    """
    Builds two lookup indexes from the file list:
      - by relative path  (exact match)
      - by stem           (filename without extension, for fuzzy matching)
    Returns a dict: { key -> relative_path }
    """
    lookup = {}
    for f in files:
        rel = f["relative"]
        # exact relative path
        lookup[rel] = rel
        # stem only e.g. "helpers" → "flask/helpers.py"
        stem = Path(rel).stem.lower()
        if stem not in lookup:
            lookup[stem] = rel
        # dot-notation key e.g. "flask.helpers" → "flask/helpers.py"
        dot_key = rel.replace("/", ".").rsplit(".", 1)[0].lower()
        if dot_key not in lookup:
            lookup[dot_key] = rel

    return lookup


def resolve(raw_import: str, source_file: str, lookup: dict, repo_root: Path) -> str | None:
    """
    Tries to resolve a raw import string to a known relative file path.
    Returns the relative path if found, else None.
    """
    raw = raw_import.strip()

    # ── 1. Relative path imports (JS/Ruby/Shell: ./foo, ../bar) ──────────────
    if raw.startswith("."):
        source_dir = Path(source_file).parent
        candidate = (source_dir / raw).resolve()
        # Try with common extensions
        for ext in ("", ".py", ".js", ".ts", ".cs", ".java", ".go", ".rb", ".rs", ".h", ".sh"):
            full = Path(str(candidate) + ext)
            # Convert back to relative
            try:
                rel = str(full.relative_to(repo_root.resolve()))
                if rel in lookup:
                    return lookup[rel]
            except ValueError:
                pass
        return None

    # ── 2. Exact relative path in lookup ─────────────────────────────────────
    if raw in lookup:
        return lookup[raw]

    # ── 3. Dot-notation → path (Python / C# / Java) ──────────────────────────
    as_path = raw.replace(".", "/").lower()
    if as_path in lookup:
        return lookup[as_path]

    # ── 4. Stem match (last segment of import) ────────────────────────────────
    stem = raw.split(".")[-1].split("/")[-1].lower()
    if stem in lookup:
        return lookup[stem]

    return None


# ── Main logic ────────────────────────────────────────────────────────────────

def extract_dependencies(
    files: list[dict],
    repo_root: Path,
    n_lines: int = 20
) -> list[tuple[str, str]]:
    """
    For each file, reads the first n_lines, extracts imports,
    resolves them to known files, and returns a list of
    (source_relative, target_relative) tuples.
    """
    lookup = build_lookup(files)
    edges  = []
    seen   = set()

    for file_meta in files:
        rel      = file_meta["relative"]
        language = file_meta.get("language", "")
        extractor = EXTRACTORS.get(language)

        if not extractor:
            continue

        full_path = repo_root / rel
        if not full_path.exists():
            continue

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(n_lines)]
        except OSError:
            continue

        raw_imports = extractor(lines)

        for raw in raw_imports:
            target = resolve(raw, rel, lookup, repo_root)
            if target and target != rel:
                key = (rel, target)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)

    return edges


def push_edges(edges: list[tuple[str, str]], conn, repo_name: str):
    """
    Creates DEPENDS_ON edges in Kùzu, scoped to one repo via
    uid = "<repo>::<relative>" so edges never cross repositories.

    Writes in UNWIND batches rather than one query per edge — the per-edge
    approach cost one round-trip each, which dominated ingestion time.
    """
    total = len(edges)
    print(f"\n  Creating {total} DEPENDS_ON edges for repo '{repo_name}'...")

    rows = [{"s": uid_for(repo_name, s), "t": uid_for(repo_name, t)}
            for s, t in edges]

    done = 0
    for chunk in batched(rows):
        conn.execute("""
            UNWIND $rows AS row
            MATCH (a:File {uid: row.s}), (b:File {uid: row.t})
            MERGE (a)-[:DEPENDS_ON]->(b)
        """, {"rows": chunk})
        done += len(chunk)
        pct = int(done / total * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\r  [{bar}] {done}/{total}", end="", flush=True)

    stored = scalar(conn.execute(
        "MATCH (a:File {repository_name: $repo})-[r:DEPENDS_ON]->() RETURN count(r)",
        {"repo": repo_name}))
    print(f"\n  Done — {stored} DEPENDS_ON edges now in the graph.")

    if stored == 0 and total > 0:
        # MATCH ... MERGE silently does nothing when the nodes aren't found, so
        # surface it rather than reporting a successful no-op.
        n_nodes = scalar(conn.execute(
            "MATCH (f:File {repository_name: $repo}) RETURN count(f)", {"repo": repo_name}))
        print(f"\n  ⚠ Resolved {total} dependencies but stored 0 edges.")
        if n_nodes == 0:
            print(f"    No File nodes exist for repo '{repo_name}'.")
            print(f"    Run the scan stage first, and make sure its --repo matches:")
            print(f"      python3 -m kb.graph.json_to_graph <tree.json> --repo {repo_name}")
        else:
            print(f"    {n_nodes} nodes exist for '{repo_name}' but no edges matched —"
                  f" check that the file paths agree.")
    print()


def print_summary(edges: list[tuple[str, str]]):
    if not edges:
        print("  No dependencies resolved — check that --repo points to the repo root.")
        return

    # Top 10 most depended-upon files
    from collections import Counter
    targets = Counter(t for _, t in edges)
    print("  Top depended-upon files:")
    for rel, count in targets.most_common(10):
        print(f"    {count:>4}x  {rel}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Groundwork — build DEPENDS_ON edges from import statements"
    )
    parser.add_argument("json_file", help="Path to JSON file list from tree_to_json.py")
    parser.add_argument("--repo",    required=True, help="Path to the repository root on disk")
    parser.add_argument("--repo-name", default=None,
                        help="Repository NAME as stored in the graph "
                             "(default: basename of --repo). Must match json_to_graph's --repo.")
    parser.add_argument("--lines",   type=int, default=20, help="Lines to read per file (default: 20)")
    parser.add_argument("--db",      default=None,
                        help=f"Kùzu database path (default: {KUZU_DB_PATH})")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: '{json_path}' not found.")
        sys.exit(1)

    repo_root = Path(args.repo)
    if not repo_root.exists():
        print(f"Error: repo '{repo_root}' not found.")
        sys.exit(1)

    with open(json_path) as f:
        all_files = json.load(f)

    # Only process code files
    CODE_LANGUAGES = {
        "Python", "JavaScript", "TypeScript",
        "JavaScript (React)", "TypeScript (React)",
        "Java", "Kotlin", "C#", "C++", "C", "C/C++ Header",
        "Go", "Rust", "Ruby", "PHP", "Swift", "Shell", "Batch",
    }
    files = [f for f in all_files if f.get("language") in CODE_LANGUAGES]
    print(f"\n  Loaded {len(files)} code files from {json_path}")
    print(f"  Reading first {args.lines} lines per file to extract imports...")

    edges = extract_dependencies(files, repo_root, n_lines=args.lines)
    print(f"  Resolved {len(edges)} dependencies between known files.")
    print_summary(edges)

    if not edges:
        sys.exit(0)

    repo_name = args.repo_name or Path(args.repo).name
    conn = get_connection(args.db)
    push_edges(edges, conn, repo_name)

    print("  Cypher to view dependencies:")
    print("  MATCH (a:File)-[:DEPENDS_ON]->(b:File) RETURN a.name, b.name\n")


if __name__ == "__main__":
    main()