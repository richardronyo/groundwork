#!/usr/bin/env python3
"""
Groundwork — Tree to JSON
Parses `tree` CLI output (or any tree-style text) into a flat JSON file list.

Usage:
    tree /path/to/repo | python3 tree_to_json.py > files.json
    python3 tree_to_json.py < nopCommerce_tree.txt > files.json
    python3 tree_to_json.py nopCommerce_tree.txt > files.json
"""

import sys
import json
import re
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

# Matches tree drawing characters: ├── └── │   and leading spaces
TREE_PREFIX_RE = re.compile(r'^[│├└─\s]+')
# Summary line at bottom e.g. "1055 directories, 6649 files"
SUMMARY_RE = re.compile(r'^\d+ director')


def get_language(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return LANGUAGE_MAP.get(ext, "Other")


def get_depth(line: str) -> int:
    """
    Depth is determined by how many 4-char tree-prefix blocks precede the name.
    Each level adds one of: '│   ' '    ' '├── ' '└── '
    """
    prefix = TREE_PREFIX_RE.match(line)
    if not prefix:
        return 0
    return len(prefix.group(0)) // 4


def is_dir(name: str) -> bool:
    return name.endswith("/")


def parse_tree(lines: list[str], repo_root: str = "") -> list[dict]:
    files = []
    # dir_stack[depth] = current directory name at that depth
    dir_stack: list[str] = [repo_root]

    for line in lines:
        # Skip blank lines and summary lines
        stripped = line.rstrip()
        if not stripped or SUMMARY_RE.match(stripped):
            continue

        # Extract the actual name by stripping tree characters
        name = TREE_PREFIX_RE.sub("", stripped).strip()
        if not name:
            continue

        depth = get_depth(stripped)

        if is_dir(name):
            # Directory — push onto stack at this depth
            clean = name.rstrip("/")
            # Trim stack to current depth and append
            dir_stack = dir_stack[: depth + 1]
            if len(dir_stack) <= depth:
                dir_stack.append(clean)
            else:
                dir_stack[depth] = clean
        else:
            # File — build its relative path from the stack
            parent_parts = dir_stack[1 : depth + 1]  # skip repo root
            relative = "/".join(parent_parts + [name]) if parent_parts else name
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

    # First non-empty line is the repo root name
    repo_root = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not SUMMARY_RE.match(stripped):
            repo_root = stripped.rstrip("/")
            lines = lines[1:]  # consume it
            break

    files = parse_tree(lines, repo_root)

    print(json.dumps(files, indent=2))

    # Print summary to stderr so it doesn't pollute the JSON pipe
    langs: dict[str, int] = {}
    for f in files:
        langs[f["language"]] = langs.get(f["language"], 0) + 1

    print(f"\n  Parsed {len(files)} files from '{repo_root}'", file=sys.stderr)
    print(  "  Languages:", file=sys.stderr)
    for lang, count in sorted(langs.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(count // 10 + 1, 30)
        print(f"    {lang:<28} {bar} {count}", file=sys.stderr)


if __name__ == "__main__":
    main()