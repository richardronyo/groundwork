#!/usr/bin/env python3
"""
Groundwork — Tree to JSON

Parses `tree -fi <repo_path>` output into a flat JSON file list. `relative`
always starts with the repo name itself, e.g. "nopCommerce/src/Libraries/...".

INVOCATION CHANGED: this expects `tree -fi`, not `tree -f`.
    -f  full path per entry (unchanged)
    -i  no indentation / box-drawing prefix — one full path per line, nothing
        else to parse

Why the rewrite: the previous version parsed `tree -f`'s indentation to
reconstruct the directory hierarchy, using an `is_dir()` check that looked
for a trailing "/" on directory names. `tree -f` alone never adds that
(only `-F` does), so the directory branch never fired, `relative` never
sees anything past the initial repo_root, and every entry's `relative`
ended up being the RAW FULL ABSOLUTE PATH from disk. That's what ended up
sitting in Kùzu. Parsing `-fi`'s flat full-path-per-line output instead
sidesteps indentation entirely — no dependency on tree's Unicode-vs-ASCII
connector characters, which vary by locale and could break the old parser
outright in a non-UTF-8 shell.

Usage:
    tree -fi /path/to/repo | python3 tree_to_json.py > files.json
    python3 tree_to_json.py nopCommerce_tree.txt > files.json   # from a saved -fi dump
"""

import sys
import json
import os

LANGUAGE_MAP = {
    "py":       "Python",
    "js":       "JavaScript",
    "ts":       "TypeScript",
    "jsx":      "JavaScript (React)",
    "tsx":      "TypeScript (React)",
    "java":     "Java",
    "kt":       "Kotlin",
    "cs":       "C#",
    "cpp":      "C++",
    "cc":       "C++",
    "cxx":      "C++",
    "c":        "C",
    "h":        "C/C++ Header",
    "hpp":      "C/C++ Header",
    "go":       "Go",
    "rs":       "Rust",
    "rb":       "Ruby",
    "php":      "PHP",
    "swift":    "Swift",
    "md":       "Markdown",
    "markdown": "Markdown",
    "json":     "JSON",
    "yaml":     "YAML",
    "yml":      "YAML",
    "toml":     "TOML",
    "html":     "HTML",
    "htm":      "HTML",
    "css":      "CSS",
    "scss":     "SCSS",
    "sass":     "SCSS",
    "sql":      "SQL",
    "sh":       "Shell",
    "bash":     "Shell",
    "zsh":      "Shell",
    "bat":      "Batch",
    "cmd":      "Batch",
    "csproj":   "MSBuild",
    "sln":      "MSBuild",
    "props":    "MSBuild",
    "targets":  "MSBuild",
    "cshtml":   "Razor",
    "razor":    "Razor",
    "resx":     "Resource",
    "xml":      "XML",
}


def get_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return LANGUAGE_MAP.get(ext, "Other")


def parse_tree(lines: list[str], repo_root: str, repo_name: str) -> list[dict]:
    """
    Each line is a full absolute path (tree -fi output — one entry per line,
    no indentation to parse). Directories are skipped: json_to_graph.py
    already reconstructs the directory hierarchy from each file's `relative`
    path, so there's nothing to lose by not emitting them here.

    `relative` is built by stripping repo_root's own path off the front and
    re-prefixing with repo_name, so it always reads "<repo_name>/...".
    """
    files = []
    prefix = repo_root.rstrip("/") + "/"

    for line in lines:
        full_path = line.rstrip("\n").rstrip()
        if not full_path or not full_path.startswith(prefix):
            continue          # blank line or stray non-path output (e.g. a report line)

        if os.path.isdir(full_path):
            continue          # directories are derived downstream from file paths

        tail = full_path[len(prefix):]        # path within the repo, no repo name yet
        relative = f"{repo_name}/{tail}"
        name = os.path.basename(full_path)
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        files.append({
            "name":      name,
            "relative":  relative,
            "extension": ext,
            "language":  get_language(name),
        })

    return files


def main():
    # Accept optional filename arg; otherwise read stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    if not lines:
        print("[]")
        return

    # First non-empty line is the repo root's own full path (tree -fi prints it bare)
    repo_root = ""
    for line in lines:
        stripped = line.strip()
        if stripped:
            repo_root = stripped.rstrip("/")
            lines = lines[1:]  # consume it
            break

    repo_name = os.path.basename(repo_root)
    files = parse_tree(lines, repo_root, repo_name)

    print(json.dumps(files, indent=2))

    # Print summary to stderr so it doesn't pollute the JSON pipe
    langs: dict[str, int] = {}
    for f in files:
        langs[f["language"]] = langs.get(f["language"], 0) + 1

    print(f"\n  Parsed {len(files)} files from '{repo_name}'", file=sys.stderr)
    print(  "  Languages:", file=sys.stderr)
    for lang, count in sorted(langs.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(count // 10 + 1, 30)
        print(f"    {lang:<28} {bar} {count}", file=sys.stderr)


if __name__ == "__main__":
    main()