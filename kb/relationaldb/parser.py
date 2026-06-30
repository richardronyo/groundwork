#!/usr/bin/env python3
"""
Groundwork — Repository Parser
Extracts per-file metrics: language, lines, classes, functions, methods, imports.
Python uses the `ast` module; other languages get a line count only.

Library:
    from parser import analyze_repo
    results = analyze_repo("./flask")   # { rel_path: {language, metrics} }

CLI:
    python3 parser.py ./flask
"""

import ast
import json
import sys
import argparse
from pathlib import Path


LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "JavaScript (React)", ".tsx": "TypeScript (React)",
    ".java": "Java", ".cs": "C#", ".cpp": "C++", ".c": "C",
    ".go": "Go", ".rb": "Ruby", ".rs": "Rust", ".php": "PHP", ".swift": "Swift",
}

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv",
               "venv", "dist", "build", "bin", "obj", ".vs", ".idea"}


def detect_language(path: Path) -> str:
    return LANGUAGE_MAP.get(path.suffix.lower(), "Other")


class MetricsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = self.functions = self.methods = 0
        self.async_functions = self.imports = self.class_depth = 0

    def visit_ClassDef(self, node):
        self.classes += 1
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_FunctionDef(self, node):
        if self.class_depth:
            self.methods += 1
        else:
            self.functions += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.async_functions += 1
        if self.class_depth:
            self.methods += 1
        else:
            self.functions += 1
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports += len(node.names)

    def visit_ImportFrom(self, node):
        self.imports += len(node.names)


def _empty_metrics(lines=0):
    return {"classes": 0, "functions": 0, "methods": 0,
            "async_functions": 0, "imports": 0, "lines": lines}


def analyze_python(path: Path):
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        v = MetricsVisitor()
        v.visit(tree)
        return {
            "classes": v.classes, "functions": v.functions, "methods": v.methods,
            "async_functions": v.async_functions, "imports": v.imports,
            "lines": len(source.splitlines()),
        }
    except Exception as e:
        print(f"  Failed to analyze {path}: {e}")
        return None


def analyze_generic(path: Path):
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        return _empty_metrics(len(source.splitlines()))
    except Exception as e:
        print(f"  Failed to read {path}: {e}")
        return None


def analyze_file(path: Path):
    language = detect_language(path)
    metrics = analyze_python(path) if language == "Python" else analyze_generic(path)
    if metrics is None:
        return None
    return {"language": language, "metrics": metrics}


def analyze_repo(repo_root: str, extensions: tuple = None) -> dict:
    """Walks the repo, returns { relative_path: {language, metrics} }."""
    repo_root = Path(repo_root)
    results = {}
    patterns = extensions if extensions else tuple(LANGUAGE_MAP.keys())

    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in patterns:
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        analysis = analyze_file(path)
        if analysis:
            results[str(path.relative_to(repo_root))] = analysis
    return results


def main():
    parser = argparse.ArgumentParser(description="Groundwork — repository parser")
    parser.add_argument("repo")
    parser.add_argument("--python-only", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()

    repo_path = Path(args.repo)
    if not repo_path.exists():
        print(f"Error: '{repo_path}' not found.")
        sys.exit(1)

    extensions = (".py",) if args.python_only else None
    print(f"\n  Analyzing {repo_path}...")
    results = analyze_repo(args.repo, extensions=extensions)
    print(f"  Found {len(results)} source files.\n")

    by_lang, total_lines = {}, 0
    for data in results.values():
        by_lang[data["language"]] = by_lang.get(data["language"], 0) + 1
        total_lines += data["metrics"]["lines"]

    print("  By language:")
    for lang, count in sorted(by_lang.items(), key=lambda x: -x[1]):
        print(f"    {lang:<20} {count}")
    print(f"\n  Total lines: {total_lines}\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Written to {args.json}\n")


if __name__ == "__main__":
    main()