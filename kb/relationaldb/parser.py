#!/usr/bin/env python3
"""
Groundwork — Repository Parser
Extracts per-file metrics (language, lines, classes, functions, methods,
imports) AND structural detail — the actual names, base classes, attributes,
methods, and free functions, not just counts.

Python is exact, via the `ast` module, and handles nested classes/functions
correctly (a method stays a method even if its class is itself nested).
C#, JavaScript, and TypeScript get structural detail too — classes with
attributes and methods — via regex + brace-counting; there's no real parser
for them here, so treat those three as best-effort (same approach and same
tradeoff kb/diagram.py's UML class scanner already makes, kept consistent
with it on purpose). Free-function extraction (module-level, not inside a
class) is Python-only for now. Every other language still gets a line count
only, same as before.

Library:
    from parser import analyze_repo
    results = analyze_repo("./flask")
    # { rel_path: {language, metrics, structure} }
    # structure = {"classes": [...], "functions": [...]}  (see StructureVisitor
    # / _scan_csharp_structure / _scan_js_structure for the exact shape)

CLI:
    python3 parser.py ./flask
"""

import ast
import json
import re
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
        # A def nested inside another def (a local helper/closure) isn't its
        # own method or function — without tracking this separately from
        # class_depth, a closure built inside __init__ (e.g. a callback
        # passed to something) gets counted as a sibling method, since
        # class_depth stays > 0 the whole time regardless of function
        # nesting. This previously inflated `methods` by one for every such
        # closure with no way to notice, since only the count existed.
        self.func_depth = 0

    def visit_ClassDef(self, node):
        self.classes += 1
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_FunctionDef(self, node):
        if self.func_depth == 0:
            if self.class_depth:
                self.methods += 1
            else:
                self.functions += 1
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        if self.func_depth == 0:
            self.async_functions += 1
            if self.class_depth:
                self.methods += 1
            else:
                self.functions += 1
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    def visit_Import(self, node):
        self.imports += len(node.names)

    def visit_ImportFrom(self, node):
        self.imports += len(node.names)


# ── Structural extraction (names, not just counts) ────────────────────────────

STRUCTURED_LANGUAGES = {"Python", "C#", "JavaScript", "JavaScript (React)",
                        "TypeScript", "TypeScript (React)"}


def _empty_structure() -> dict:
    return {"classes": [], "functions": []}


def _py_expr(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


class StructureVisitor(ast.NodeVisitor):
    """
    Like MetricsVisitor, but collects names/signatures instead of just
    counts. A class stack tracks the innermost enclosing class, the same way
    MetricsVisitor tracks class_depth — so a method stays attached to its
    class (and a class nested inside another class is still found) while
    only genuinely module-level defs land in `functions`.
    """

    def __init__(self):
        self.classes: list[dict] = []
        self.functions: list[dict] = []
        self._class_stack: list[dict] = []
        self._seen_attrs: list[set] = []
        self._func_depth = 0    # >0 means we're inside another def's body already

    def visit_ClassDef(self, node):
        bases = [_py_expr(b) for b in node.bases if _py_expr(b)]
        entry = {"name": node.name, "bases": bases, "attrs": [], "methods": []}
        self.classes.append(entry)
        self._class_stack.append(entry)
        self._seen_attrs.append(set())
        self.generic_visit(node)
        self._class_stack.pop()
        self._seen_attrs.pop()

    def _visit_func(self, node, is_async: bool):
        # A def nested inside another def (a local helper/closure, like a
        # callback built inside __init__) is not a method of the enclosing
        # class — it belongs to whichever function it's actually local to,
        # which we don't track as a first-class thing. Only defs found
        # directly in a class body or at module level count.
        top_level = self._func_depth == 0
        in_class = top_level and bool(self._class_stack)
        params = [a.arg for a in node.args.args if not (in_class and a.arg == "self")]
        ret = _py_expr(node.returns) if node.returns else ""
        vis = "-" if node.name.startswith("_") else "+"
        entry = {"name": node.name, "params": params, "ret": ret, "vis": vis, "is_async": is_async}
        if top_level:
            if in_class:
                self._class_stack[-1]["methods"].append(entry)
                if node.name == "__init__":
                    self._collect_self_attrs(node)
            else:
                self.functions.append(entry)
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_FunctionDef(self, node):
        self._visit_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node, is_async=True)

    def _collect_self_attrs(self, init_node):
        seen = self._seen_attrs[-1]
        cls = self._class_stack[-1]
        for sub in ast.walk(init_node):
            target, ann = None, ""
            if (isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Attribute)
                    and isinstance(sub.target.value, ast.Name) and sub.target.value.id == "self"):
                target, ann = sub.target.attr, _py_expr(sub.annotation)
            elif (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store)
                    and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
                target = sub.attr
            if target and target not in seen:
                seen.add(target)
                cls["attrs"].append({"name": target, "type": ann,
                                    "vis": "-" if target.startswith("_") else "+"})

    def visit_AnnAssign(self, node):
        # Class-level annotated assignment, e.g. `x: int` directly in a class
        # body (dataclass style) — self.x inside __init__ is handled above.
        if self._class_stack and isinstance(node.target, ast.Name):
            nm = node.target.id
            seen = self._seen_attrs[-1]
            if nm not in seen:
                seen.add(nm)
                self._class_stack[-1]["attrs"].append(
                    {"name": nm, "type": _py_expr(node.annotation),
                     "vis": "-" if nm.startswith("_") else "+"})
        self.generic_visit(node)


def _scan_python_structure(tree: ast.Module) -> dict:
    v = StructureVisitor()
    v.visit(tree)
    return {"classes": v.classes, "functions": v.functions}


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


def _scan_csharp_structure(text: str) -> dict:
    text = _strip_comments(text, "C#")
    classes = []
    for m in CS_CLASS_HEAD.finditer(text):
        name = m.group(1)
        bases = [b.strip() for b in (m.group(2) or "").split(",") if b.strip()]
        body, _ = _extract_braced_body(text, m.end() - 1)

        methods = []
        for mm in CS_METHOD.finditer(body):
            mname = mm.group(1)
            if mname == name:      # constructor — same name as class, not a method
                continue
            params = [p.strip().split()[-1] for p in mm.group(2).split(",") if p.strip()]
            methods.append({"name": mname, "params": params, "ret": "",
                           "vis": "+", "is_async": "async" in mm.group(0)})

        attrs, seen = [], set()
        for fm in CS_FIELD.finditer(body):
            ftype, fname = fm.group(1), fm.group(2)
            if fname in seen or ftype in ("class", "return", "using"):
                continue
            seen.add(fname)
            attrs.append({"name": fname, "type": ftype, "vis": "+"})

        classes.append({"name": name, "bases": bases, "attrs": attrs[:12], "methods": methods[:20]})
    return {"classes": classes, "functions": []}


JS_CLASS_HEAD = re.compile(r"class\s+(\w+)\s*(?:extends\s+([\w.]+))?\s*\{")
JS_METHOD = re.compile(r"(?:^|\n)\s*(?:static\s+|async\s+|get\s+|set\s+)*(\w+)\s*\(([^)]*)\)\s*\{")
JS_SKIP_METHOD_NAMES = {"if", "for", "while", "switch", "catch", "constructor"}


def _scan_js_structure(text: str, lang: str) -> dict:
    text = _strip_comments(text, lang)
    classes = []
    for m in JS_CLASS_HEAD.finditer(text):
        name, base = m.group(1), m.group(2)
        body, _ = _extract_braced_body(text, m.end() - 1)

        methods = []
        for mm in JS_METHOD.finditer(body):
            mname = mm.group(1)
            if mname in JS_SKIP_METHOD_NAMES:
                continue
            params = [p.strip() for p in mm.group(2).split(",") if p.strip()]
            methods.append({"name": mname, "params": params, "ret": "",
                           "vis": "+", "is_async": "async" in mm.group(0)})

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

        classes.append({"name": name, "bases": [base] if base else [],
                        "attrs": attrs[:12], "methods": methods[:20]})
    return {"classes": classes, "functions": []}


def _empty_metrics(lines=0):
    return {"classes": 0, "functions": 0, "methods": 0,
            "async_functions": 0, "imports": 0, "lines": lines}


def analyze_python(path: Path):
    """Returns (metrics, structure) or (None, None) on failure."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        v = MetricsVisitor()
        v.visit(tree)
        metrics = {
            "classes": v.classes, "functions": v.functions, "methods": v.methods,
            "async_functions": v.async_functions, "imports": v.imports,
            "lines": len(source.splitlines()),
        }
        return metrics, _scan_python_structure(tree)
    except Exception as e:
        print(f"  Failed to analyze {path}: {e}")
        return None, None


def analyze_structured(path: Path, language: str):
    """C#/JS/TS: structural detail via regex + brace-counting, and metrics
    counts DERIVED from that same structure so they're finally non-zero for
    these languages too. Returns (metrics, structure) or (None, None)."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  Failed to read {path}: {e}")
        return None, None

    structure = (_scan_csharp_structure(source) if language == "C#"
                else _scan_js_structure(source, language))

    n_methods = sum(len(c["methods"]) for c in structure["classes"])
    n_async = sum(1 for c in structure["classes"] for m in c["methods"] if m["is_async"])
    metrics = {
        "classes": len(structure["classes"]),
        "functions": len(structure["functions"]),
        "methods": n_methods,
        "async_functions": n_async,
        "imports": 0,       # not tracked for these languages
        "lines": len(source.splitlines()),
    }
    return metrics, structure


def analyze_generic(path: Path):
    """Returns (metrics, structure) or (None, None) on failure. structure is
    always empty here — this is the "line count only" fallback for languages
    with no scanner at all."""
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        return _empty_metrics(len(source.splitlines())), _empty_structure()
    except Exception as e:
        print(f"  Failed to read {path}: {e}")
        return None, None


def analyze_file(path: Path):
    language = detect_language(path)
    if language == "Python":
        metrics, structure = analyze_python(path)
    elif language in STRUCTURED_LANGUAGES:
        metrics, structure = analyze_structured(path, language)
    else:
        metrics, structure = analyze_generic(path)
    if metrics is None:
        return None
    return {"language": language, "metrics": metrics, "structure": structure}


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