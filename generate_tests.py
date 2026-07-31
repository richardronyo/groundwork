#!/usr/bin/env python3
"""
Groundwork — Unit Test Generation

Generates unit tests for a target file (or a single function within it) by
combining two sources of grounding:
  1. The actual source code (read from disk)
  2. The business rules for that file (from PostgreSQL)
  3. Its dependencies (from Kùzu) — so the LLM knows the collaborators

The test framework is inferred from the file's language (pytest for Python,
Jest for JS/TS, JUnit for Java, xUnit for C#, etc.).

Run from the PROJECT ROOT (the directory containing kb/).

Output:
    generated_tests/   the test files
    weakness/          the weakness reports, as PDF

Usage:
    python3 -m generate_tests --repo flask --file src/app.py
    python3 -m generate_tests --repo flask --file src/auth.py --function login
    python3 -m generate_tests --repo flask --file src/app.py --out tests/
    python3 -m generate_tests --repo flask --file src/app.py --print

Requirements:
    pip install openai psycopg kuzu python-dotenv
"""

import os
import re
import sys
import argparse
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from kb.relationaldb.initialize_db import (
    get_connection, list_repositories,
    load_business_rules_from_db,
)
from weakness_helper import markdown_to_pdf, timestamp

# Reuse the same retrieval helpers grab_context uses, so test generation and
# querying draw on one shared context layer.
from grab_context import (
    GroundworkRetriever,
    get_dependencies as kb_get_dependencies,
    get_business_rules as kb_get_business_rules,
    load_key_points_from_db,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = "gpt-5-mini"


# ── Framework inference ───────────────────────────────────────────────────────

FRAMEWORK_BY_EXT = {
    ".py":   ("pytest", "test_{stem}.py"),
    ".js":   ("Jest", "{stem}.test.js"),
    ".jsx":  ("Jest + React Testing Library", "{stem}.test.jsx"),
    ".ts":   ("Jest", "{stem}.test.ts"),
    ".tsx":  ("Jest + React Testing Library", "{stem}.test.tsx"),
    ".java": ("JUnit 5", "{Stem}Test.java"),
    ".cs":   ("xUnit", "{Stem}Tests.cs"),
    ".go":   ("Go testing", "{stem}_test.go"),
    ".rb":   ("RSpec", "{stem}_spec.rb"),
    ".rs":   ("Rust #[test]", "{stem}_test.rs"),
    ".php":  ("PHPUnit", "{Stem}Test.php"),
}


def infer_framework(file_path: str):
    ext = Path(file_path).suffix.lower()
    return FRAMEWORK_BY_EXT.get(ext, ("the language's standard test framework", "test_{stem}.txt"))


def output_filename(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    stem = Path(file_path).stem
    _, pattern = FRAMEWORK_BY_EXT.get(ext, (None, "test_{stem}.txt"))
    return pattern.format(stem=stem, Stem=stem[:1].upper() + stem[1:])


# ── Existing-test detection ────────────────────────────────────────────────────
#
# infer_framework() above is a pure guess from file extension (.cs -> xUnit).
# That's a reasonable default for a repo with no tests yet, but if the repo
# already HAS tests, the actually-correct answer is whatever THEY use — a
# .NET repo could be on NUnit or MSTest instead of xUnit, a JS repo could be
# on Vitest instead of Jest, etc. This finds the repo's own test files (by
# convention: living under a "Test(s)" path segment, or with "test" in the
# filename — no special DB flag needed) and sniffs the actual framework from
# their content, falling back to the extension guess only if nothing's found.

TEST_DIR_RE = re.compile(r"(^|/)(?:[Tt]ests?|[Ss]pecs?)(/|$)")
TEST_WORD_RE = re.compile(r"^(tests?|specs?)$")


def _stem_words(stem: str) -> list[str]:
    """Splits a filename stem into words across camelCase/PascalCase/snake_case/
    dots, e.g. 'OrderControllerTests' -> ['order','controller','tests'],
    'test_app' -> ['test','app']. Used instead of a plain substring search so
    'AttestationService' or 'ContestService' don't false-positive on "test"."""
    parts = re.split(r"[_\-.]+", stem)
    words = []
    for p in parts:
        words.extend(re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z0-9]+|[A-Z]+", p))
    return [w.lower() for w in words if w]


def _looks_like_test_file(stem: str) -> bool:
    return any(TEST_WORD_RE.match(w) for w in _stem_words(stem))

# Checked in order; first match wins. More specific variants (e.g. Jest+RTL)
# come before their more generic parent (plain Jest) since the specific one
# implies the generic one but not vice versa.
FRAMEWORK_SIGNATURES = [
    ("xUnit", re.compile(r"using\s+Xunit\s*;|\[Fact\]|\[Theory\]")),
    ("NUnit", re.compile(r"using\s+NUnit\.Framework\s*;|\[TestFixture\]")),
    ("MSTest", re.compile(r"using\s+Microsoft\.VisualStudio\.TestTools\.UnitTesting\s*;|\[TestClass\]")),
    ("pytest", re.compile(r"^\s*(?:import|from)\s+pytest\b", re.M)),
    ("unittest", re.compile(r"import\s+unittest\b|\(unittest\.TestCase\)")),
    ("Jest + React Testing Library", re.compile(r"@testing-library/react")),
    ("Jest", re.compile(r"from\s+['\"]@jest|\bjest\.(mock|fn)\(|\bdescribe\(.*\bit\(", re.S)),
    ("JUnit 5", re.compile(r"org\.junit\.jupiter")),
    ("JUnit 4", re.compile(r"import\s+org\.junit\.Test\b|import\s+org\.junit\.Assert\b")),
    ("RSpec", re.compile(r"RSpec\.describe|require\s+['\"]rspec['\"]")),
    ("Go testing", re.compile(r'"testing"[\s\S]{0,200}func\s+Test\w+\(\w+\s*\*testing\.T\)')),
    ("Rust #[test]", re.compile(r"#\[test\]|#\[cfg\(test\)\]")),
    ("PHPUnit", re.compile(r"PHPUnit\\Framework\\TestCase")),
]


def find_existing_test_files(repo_name: str) -> list[dict]:
    """
    Existing test files for a repo, found by convention: living under a path
    segment named "Test"/"Tests" (nopCommerce's own src/Tests/, for example),
    or having "test" in the filename. Returns candidates ranked with
    folder+filename matches first. Degrades to [] if Postgres is unreachable.
    """
    try:
        conn = get_connection()
    except Exception:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT file_path, language FROM files WHERE repository_name = %s",
                       (repo_name,))
            rows = cur.fetchall()
    except Exception:
        return []
    finally:
        conn.close()

    candidates = []
    for file_path, language in rows:
        in_test_dir = bool(TEST_DIR_RE.search(file_path))
        name_has_test = _looks_like_test_file(Path(file_path).stem)
        if in_test_dir or name_has_test:
            candidates.append({"file_path": file_path, "language": language,
                              "score": 2 if (in_test_dir and name_has_test) else 1})
    candidates.sort(key=lambda c: -c["score"])
    return candidates


def detect_framework_from_existing_tests(repo_name: str, repo_path: Path, rel_file: str = None,
                                         max_files_checked: int = 8):
    """
    Looks at the repo's OWN existing test files and sniffs which framework is
    actually in use from their content, rather than guessing from the
    target's extension. Returns (framework, example_file_path, example_source)
    — example_source is that file's full content, meant to be shown to the
    LLM as a concrete style reference (naming, mocking, assertion style).
    All three are None if nothing could be determined; the caller falls back
    to infer_framework()'s extension guess in that case.
    """
    if repo_path is None:
        return None, None, None

    candidates = find_existing_test_files(repo_name)
    if rel_file:
        # Prefer test files in the SAME language as the target — a repo can
        # mix languages (e.g. a C# backend with JS frontend tests).
        ext = Path(rel_file).suffix.lower()
        same_ext = [c for c in candidates if Path(c["file_path"]).suffix.lower() == ext]
        if same_ext:
            candidates = same_ext

    checked = 0
    for c in candidates:
        if checked >= max_files_checked:
            break
        # file_path is repo-name-prefixed; repo_path already ends in the repo
        # name, so join against its PARENT — same convention used throughout.
        full_path = repo_path.parent / c["file_path"]
        try:
            text = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        checked += 1
        for name, pattern in FRAMEWORK_SIGNATURES:
            if pattern.search(text):
                return name, c["file_path"], text
    return None, None, None


# ── Source + context readers ──────────────────────────────────────────────────

def read_source(repo_path: Path, rel_file: str) -> str:
    # rel_file is repo-name-prefixed (e.g. "nopCommerce/src/.../Foo.cs") —
    # matches Postgres's files.file_path and what find_top_file_for_prompt
    # returns from ChromaDB metadata. repo_path already ends in the repo
    # name, so join against its PARENT to avoid doubling it — same
    # convention as everywhere else in this project (diagram.py,
    # file_dependencies.py, extract_business_rules.py, system_report.py).
    full = repo_path.parent / rel_file
    if not full.is_file():
        sys.exit(f"Error: file not found on disk: {full}")
    return full.read_text(encoding="utf-8", errors="ignore")


def extract_function(source: str, func_name: str, language_ext: str) -> str:
    """
    Best-effort extraction of a single function/method body from source.
    For Python we use indentation; for brace languages we balance braces.
    Falls back to the whole file if the function can't be isolated.
    """
    if language_ext == ".py":
        lines = source.splitlines()
        out, capturing, indent = [], False, None
        pat = re.compile(rf"^(\s*)(async\s+)?def\s+{re.escape(func_name)}\b")
        for line in lines:
            m = pat.match(line)
            if m and not capturing:
                capturing = True
                indent = len(m.group(1))
                out.append(line)
                continue
            if capturing:
                if line.strip() == "":
                    out.append(line)
                    continue
                cur_indent = len(line) - len(line.lstrip())
                if cur_indent <= indent:
                    break
                out.append(line)
        return "\n".join(out) if out else source

    # Brace languages: find the signature, then balance { }
    idx = source.find(func_name)
    if idx == -1:
        return source
    brace = source.find("{", idx)
    if brace == -1:
        return source
    depth, i = 0, brace
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                # back up to the start of the signature line
                line_start = source.rfind("\n", 0, idx) + 1
                return source[line_start:i + 1]
        i += 1
    return source


# ── Dependency signature grounding ─────────────────────────────────────────────
#
# The compiler errors this was built to prevent were never a "the LLM is bad
# at C#" problem — they're a "the LLM was never shown the real signature"
# problem. Nothing in the old prompt gave it the actual method signatures for
# whatever it was about to mock, so for anything with more than a couple of
# parameters it just invented a plausible-looking argument list (wrong count,
# wrong order, wrong types). kb.relationaldb.metadata.py now indexes real
# classes/methods (not just counts) into PostgreSQL, so this pulls the exact
# signatures for whatever the target's constructor actually injects and
# grounds the prompt in them.

CTOR_RE = re.compile(r"public\s+\w+\s*\(([^)]*)\)", re.S)


def extract_constructor_types(source: str) -> list[str]:
    """
    Best-effort: the TYPE names from what looks like the primary constructor
    in `source` (matches typical DI-style 'public Foo(TypeA a, TypeB b)').
    These are exactly what a generated test will need to Mock<T> and Setup(...),
    so looking their real signatures up by name is far more targeted than
    walking the file-level dependency graph.
    """
    m = CTOR_RE.search(source)
    if not m:
        return []
    params = m.group(1)
    # Strip generic type args first (IAttributeParser<A, B> -> IAttributeParser)
    # so a generic's internal commas don't fragment the parameter list.
    depth, cleaned = 0, []
    for ch in params:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth -= 1
            continue
        if depth == 0:
            cleaned.append(ch)
    types = re.findall(r"([A-Z]\w*)\s+\w+\s*(?:,|$)", "".join(cleaned))
    return list(dict.fromkeys(types))  # dedupe, keep first-seen order


def get_signatures_by_name(type_names: list[str], repo_name: str, max_methods: int = 25) -> dict:
    """
    Real method signatures for classes/interfaces named in `type_names`,
    scoped to one repo, straight from PostgreSQL. Returns
    { file_path: [{"class": name, "methods": ["Foo(a, b): ret", ...]}] }.
    Degrades to {} (not an exception) if Postgres is unreachable or nothing's
    indexed yet — the caller falls back to the old deps-only context in that
    case, same convention kb/diagram.py uses for its own DB-first lookups.
    """
    if not type_names:
        return {}
    try:
        conn = get_connection()
    except Exception:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT f.file_path, c.name, c.id
                FROM classes c JOIN files f ON f.id = c.file_id
                WHERE f.repository_name = %s AND c.name = ANY(%s)
                ORDER BY f.file_path, c.name
            """, (repo_name, type_names))
            class_rows = cur.fetchall()
            if not class_rows:
                return {}

            class_ids = [r[2] for r in class_rows]
            cur.execute("""
                SELECT class_id, name, params, return_type, is_async
                FROM functions WHERE class_id = ANY(%s) ORDER BY id
            """, (class_ids,))
            method_rows = cur.fetchall()
    except Exception:
        return {}
    finally:
        conn.close()

    methods_by_class = defaultdict(list)
    for class_id, name, params, ret, is_async in method_rows:
        sig = f"{name}({', '.join(params or [])})"
        if ret:
            sig += f": {ret}"
        methods_by_class[class_id].append(("async " if is_async else "") + sig)

    result = defaultdict(list)
    for file_path, cname, cid in class_rows:
        methods = methods_by_class.get(cid, [])[:max_methods]
        if methods:   # skip entries with nothing indexed — no point listing an empty class
            result[file_path].append({"class": cname, "methods": methods})
    return dict(result)


def format_signatures_block(sig_map: dict) -> str:
    if not sig_map:
        return ("(none indexed — run `python3 -m kb.relationaldb.metadata <repo path>` on this "
                "repo to enable exact-signature grounding; until then, mocked calls are a guess)")
    lines = []
    for file_path, classes in sig_map.items():
        for c in classes:
            lines.append(f"{c['class']} ({file_path}):")
            for m in c["methods"]:
                lines.append(f"    {m}")
    return "\n".join(lines)


def read_dependency_sources(type_names: list[str], repo_name: str, repo_path: Path,
                            max_chars: int = 4000) -> dict:
    """
    The actual on-disk source for each constructor-injected dependency type —
    found in PostgreSQL, read from disk. This is the strongest grounding
    available: the model sees the REAL using directives, REAL namespace, and
    REAL signatures verbatim, rather than a synthesized summary. Added
    specifically because signatures alone still left the model guessing at
    `using` statements for referenced types, producing CS0234/CS0246 errors.

    Truncated per file (nopCommerce service interfaces can be large) — this
    trades completeness for staying within a sane prompt size when a
    constructor injects a couple dozen dependencies. Degrades to {} (not an
    exception) if Postgres is unreachable, nothing's indexed, or repo_path is
    unavailable — the caller falls back to the signature-only block already
    added above.
    """
    if not type_names or repo_path is None:
        return {}
    try:
        conn = get_connection()
    except Exception:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT f.file_path
                FROM classes c JOIN files f ON f.id = c.file_id
                WHERE f.repository_name = %s AND c.name = ANY(%s)
            """, (repo_name, type_names))
            file_paths = [r[0] for r in cur.fetchall()]
    except Exception:
        return {}
    finally:
        conn.close()

    sources = {}
    for fp in file_paths:
        # fp is repo-name-prefixed (e.g. "nopCommerce/src/.../IOrderService.cs");
        # repo_path already ends in the repo name, so join against its PARENT
        # to avoid doubling it — same convention used throughout this project.
        full_path = repo_path.parent / fp
        try:
            text = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) > max_chars:
            omitted = len(text) - max_chars
            text = text[:max_chars] + f"\n// ... truncated, {omitted} more characters"
        sources[fp] = text
    return sources


def format_dependency_sources_block(sources: dict) -> str:
    if not sources:
        return "(none available — either nothing indexed yet, or the repo isn't on disk at --repo-path)"
    return "\n\n".join(f"// ── {fp} ──\n{text}" for fp, text in sources.items())


def load_manual_example(path_str: str):
    """
    Reads a user-specified existing test file from disk — via --example-file
    on the CLI, or the TUI's Tests tab "Example test file" field — instead of
    (or overriding) the automatic search. Detects its framework with the same
    signatures auto-detection uses. Returns (framework_or_None, path_str,
    source_or_None); framework/source are None if the file couldn't be read,
    but path_str is always returned so the caller can report what was tried.
    """
    p = Path(path_str).expanduser()
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        print(f"  ⚠ Could not read --example-file '{path_str}': {e}")
        return None, str(p), None

    for name, pattern in FRAMEWORK_SIGNATURES:
        if pattern.search(text):
            return name, str(p), text
    return None, str(p), text  # readable, but no known framework signature matched


# ── Related context (shared with grab_context) ────────────────────────────────

def gather_related_context(repo_name, rel_file, target_source, repo_path=None, top_n=3):
    """
    Builds the same kind of multi-store context grab_context produces, focused
    on the target file:
      - dependency files + THEIR business rules (contracts of collaborators)
      - repo-level key points (overall system purpose)
      - semantically related files via the GroundworkRetriever (BERTScore →
        ChromaDB), seeded by the target file's own rules/source

    Returns a dict of preformatted text blocks for the prompt.
    """
    # 1. Direct dependencies of the target file, and their rules
    deps_map = kb_get_dependencies([rel_file], repo_name)
    dep_files = deps_map.get(rel_file, [])
    dep_rules = kb_get_business_rules(dep_files, repo_name) if dep_files else {}

    dep_lines = []
    for dep in dep_files:
        rules = dep_rules.get(dep, [])
        dep_lines.append(f"- {dep}")
        for r in rules[:4]:  # cap so the prompt stays focused
            dep_lines.append(f"    · {r}")
    deps_block = "\n".join(dep_lines) if dep_lines else "(none recorded)"

    # 1b. Real signatures for whatever the target's constructor actually
    #     injects — looked up by NAME, not via the file dependency graph
    #     (which is only best-effort for C#), so this works even when a
    #     dependency edge didn't resolve cleanly.
    injected_types = extract_constructor_types(target_source)
    sig_map = get_signatures_by_name(injected_types, repo_name)
    signatures_block = format_signatures_block(sig_map)

    dep_sources = read_dependency_sources(injected_types, repo_name, repo_path)
    dep_sources_block = format_dependency_sources_block(dep_sources)

    # 2. Repo-level key points — what the whole system does
    key_points = load_key_points_from_db(repo_name)
    kp_block = "\n".join(f"- {kp}" for kp in key_points[:15]) if key_points else "(none recorded)"

    # 3. Semantically related files via the shared retriever.
    #    Seed the query with the target file's rules (fallback: a snippet of source).
    related_block = "(none found)"
    try:
        retriever = GroundworkRetriever(top_n=top_n, repo_name=repo_name, mode="file")
        seed = f"tests and behavior of {rel_file}"
        docs = retriever.invoke(seed)
        related = []
        for d in docs:
            rel = d.metadata.get("relative")
            if rel and rel != rel_file:  # don't echo the target back
                related.append(f"- {rel} (score {d.metadata.get('score', 0):.3f})")
        if related:
            related_block = "\n".join(related)
    except Exception as e:
        related_block = f"(retrieval unavailable: {e})"

    return {
        "deps_block": deps_block,
        "signatures_block": signatures_block,
        "dep_sources_block": dep_sources_block,
        "key_points_block": kp_block,
        "related_block": related_block,
        "dep_count": len(dep_files),
        "kp_count": len(key_points),
    }


# ── Prompt + LLM ──────────────────────────────────────────────────────────────

# What "mark this expected to fail" actually means varies a lot by framework,
# and most of them have NOTHING like pytest's xfail. The previous version of
# this prompt only gave a pytest example and left the model to improvise for
# everything else — for xUnit it improvised [Fact(Skip = "...")], which reads
# as "expected to fail" but is really "never run this test again." A skipped
# test gives zero signal when the underlying bug is eventually fixed. Only
# pytest and RSpec have a genuine xfail (runs, expected to fail, and — with
# strict mode / by default respectively — FAILS the suite if it unexpectedly
# passes). Jest 29+'s test.failing is the same idea. Everything else gets the
# same fallback: write the test to assert the CORRECT behavior so it genuinely
# fails right now, tag it instead of skipping it, and keep it running.
XFAIL_MECHANISM = {
    "pytest": (
        'Mark it with @pytest.mark.xfail(reason="...", strict=True) directly above the test. '
        'strict=True makes an unexpected PASS fail the suite too, so a real fix is never missed.'
    ),
    "Jest": (
        'Use test.failing("name", async () => { ... }) (Jest 29+) instead of test(...). '
        'test.failing REQUIRES the test to fail — Jest itself flags it as broken once the '
        'underlying bug is fixed and the test starts passing, so nothing goes unnoticed.'
    ),
    "Jest + React Testing Library": (
        'Use test.failing("name", async () => { ... }) (Jest 29+) instead of test(...). '
        'test.failing REQUIRES the test to fail — Jest itself flags it as broken once the '
        'underlying bug is fixed and the test starts passing, so nothing goes unnoticed.'
    ),
    "JUnit 5": (
        'JUnit has no real xfail. Do NOT use @Disabled — it hides the test forever, even after '
        'the bug is fixed. Write the test to assert the CORRECT behavior (so it genuinely fails '
        'right now) and tag it @Tag("known-issue") instead. The required CI gate should exclude '
        'that tag; a separate non-blocking job runs only that tag, so a fix becomes visible the '
        "moment the test starts passing."
    ),
    "xUnit": (
        'xUnit has no real xfail. Do NOT use [Fact(Skip = "...")] — it hides the test forever, '
        'even after the bug is fixed. Write the test to assert the CORRECT behavior (so it '
        'genuinely fails right now) and use a plain [Fact] with [Trait("Category", "KnownIssue")] '
        'instead — no Skip. The required CI gate should run with '
        '--filter "Category!=KnownIssue"; a separate non-blocking job runs '
        '--filter "Category=KnownIssue", so a fix becomes visible the moment the test starts passing.'
    ),
    "Go testing": (
        'Go has no xfail, and t.Skip() has the same problem as [Fact(Skip=...)] — it hides the '
        'test forever. Write the test to assert the CORRECT behavior (so it genuinely fails now), '
        'name it TestKnownIssue_... instead of Test..., and add a "// KNOWN ISSUE: reason" comment '
        'above it, so it can be excluded from the required run with -run "^Test_" while remaining '
        'discoverable and runnable on its own.'
    ),
    "RSpec": (
        'Use pending("reason") as the first line inside the test (or '
        '`it "...", pending: "reason" do ... end`). This is a real xfail: RSpec runs the test, '
        'reports it as pending (not a failure) if it fails as expected, and — critically — FAILS '
        'the suite if a pending test unexpectedly passes, so a fix is never silently missed.'
    ),
    "Rust #[test]": (
        'Rust has no xfail. Do NOT use #[ignore] for this — it hides the test forever unless '
        'someone remembers `cargo test -- --ignored`. Write the test to assert the CORRECT '
        'behavior (so it genuinely fails now) and put a "// KNOWN ISSUE: reason" comment directly '
        'above it, so it shows up in source and in CI output when it fails.'
    ),
    "PHPUnit": (
        "PHPUnit has no real xfail — do NOT use markTestSkipped() or markTestIncomplete(), both "
        "hide the test forever. Write the test to assert the CORRECT behavior (so it genuinely "
        "fails now) and add #[Group('known-issue')] above it. The required CI gate should run "
        "--exclude-group known-issue; a separate non-blocking job runs --group known-issue, so a "
        "fix becomes visible the moment the test starts passing."
    ),
}
DEFAULT_XFAIL_MECHANISM = (
    "This language/framework has no known xfail mechanism. Write the test to assert the CORRECT "
    'behavior (so it genuinely fails right now) and add a clear "// KNOWN ISSUE: reason" comment '
    "(or that language's equivalent) directly above it, rather than skipping or disabling it — a "
    "skipped test gives no signal at all when the underlying bug is eventually fixed."
)

TEST_SYSTEM = """You are a senior test engineer who specializes in finding bugs.
You write thorough, idiomatic unit tests whose goal is to EXPOSE WEAKNESSES:
missing validation, unhandled inputs, boundary and overflow conditions, error
paths that aren't handled, and business rules the code fails to enforce.

You write two categories of tests:
  1. Tests that PASS against the current code (documenting correct behavior).
  2. Tests that PROBE suspected weaknesses. When you believe the current code
     would fail or behave wrongly for an input, still write the test asserting
     the CORRECT behavior — it should genuinely fail right now — and mark it
     as expected-to-fail using THIS framework's specific mechanism:

     {xfail_mechanism}

     Never use a plain skip/disable mechanism for this (Skip, @Disabled,
     test.skip, markTestSkipped, #[ignore], etc.) unless that IS the
     mechanism named above — skipping removes the test from the run
     entirely, so it gives no signal at all when the bug is eventually fixed.
     The test must actually execute.

You only reference APIs that exist in the given source — you never invent them."""

TEST_PROMPT = """Write unit tests for the target below using {framework}.

TARGET: {target_desc}
REPOSITORY: {repo}
FILE: {file}

--- SOURCE CODE ---
{source}

--- BUSINESS RULES FOR THIS FILE (each should map to at least one test) ---
{rules}

--- DEPENDENCIES + THEIR BUSINESS RULES (collaborators to mock; use their rules
    to understand the contracts this file relies on) ---
{deps}

--- REAL METHOD SIGNATURES FOR THOSE DEPENDENCIES (from the indexed source —
    use these for any Setup(...)/Verify(...)/callback on a mock; if a method
    you need isn't listed here, it wasn't indexed — do not invent its
    signature, and prefer It.IsAny<T>() only for parameters shown below) ---
{signatures}

--- ACTUAL SOURCE OF THOSE DEPENDENCIES, VERBATIM FROM DISK (ground truth for
    `using`/import statements and namespaces — copy them from here rather
    than guessing; some files may be truncated if very large) ---
{dep_sources}

--- EXISTING TEST FILE FROM THIS REPO, FOR STYLE ---
{example}

--- REPOSITORY KEY POINTS (overall system purpose, for context) ---
{key_points}

--- RELATED FILES (retrieved from the knowledge base; may share behavior) ---
{related}

Requirements:
- Use {framework} idioms and conventions.
- If EXISTING TEST FILE FROM THIS REPO is present above (not the "none found"
  placeholder), match its conventions: test naming pattern, mocking library
  and setup style, assertion style, and file/class structure. Consistency
  with the repo's own existing tests matters more than any "ideal" pattern
  you might otherwise prefer. This INCLUDES using the same test framework as
  that example (e.g. if it uses NUnit, use NUnit — not xUnit or MSTest), and
  you MUST include that framework's own using/import statement (e.g.
  `using NUnit.Framework;` for NUnit, `using Xunit;` for xUnit, `import
  pytest` for pytest) — using a framework's attributes/decorators without
  importing the framework itself doesn't compile.
- Cover happy paths, then aggressively probe weaknesses: malformed/empty/None
  inputs, boundary and overflow values, wrong types, division-by-zero, unhandled
  exceptions, and any business rule the code may NOT actually enforce.
- Turn each business rule for THIS file into at least one explicit test, named so
  the rule it verifies is obvious. If you suspect the code violates a rule, assert
  the CORRECT behavior and mark the test expected-to-fail with a reason.
- For every test that probes a suspected weakness, mark it expected-to-fail using
  the framework's mechanism and give a reason naming the weakness.
- Use the dependencies' rules to mock collaborators faithfully.
- Every mocked call (Setup, Verify, It.Is<T>, callback parameter lists) on a type
  listed in REAL METHOD SIGNATURES must match that signature EXACTLY — same
  parameter count, order, and types. Never invent, reorder, add, or drop a
  parameter, and never guess a signature for a method that isn't listed there.
- Every `using`/import statement in your output for a type shown in ACTUAL SOURCE
  OF THOSE DEPENDENCIES must match the namespace that file actually declares —
  copy it, don't reconstruct it from the file path or the type's name.
- When calling a constructor or method with many parameters, pass arguments
  POSITIONALLY, in the exact order shown in the source — do not switch to
  named-argument syntax (e.g. C#'s `paramName: value`) partway through a call
  as a way to "keep track" of a long argument list, and never invent a label
  that isn't the real declared parameter name. If you use named arguments at
  all, use them for every argument in that call, spelled exactly as declared,
  and never reuse or repeat a label.
- Mock external dependencies where appropriate (DB, network, filesystem).
- Include necessary imports and any fixtures/setup.
- Output ONLY the test file contents — no prose, no markdown fences."""


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        sys.exit("Error: OPENAI_API_KEY not set in .env or environment.")
    return OpenAI(api_key=OPENAI_API_KEY)


def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        # drop first fence line and a trailing fence if present
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


# ── Weakness analysis ─────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a senior code reviewer performing a defensive audit.
You identify concrete weaknesses, uncovered cases, and likely bugs in the given
code, cross-checking it against its documented business rules. You are specific
and practical: every finding names a real input or scenario, and every finding
comes with an actionable fix. You do not pad the report with generic advice."""

ANALYSIS_PROMPT = """Audit the target below and produce a WEAKNESSES REPORT in Markdown.

TARGET: {target_desc}
FILE: {file}

--- SOURCE CODE ---
{source}

--- BUSINESS RULES FOR THIS FILE ---
{rules}

--- TESTS JUST GENERATED (some may be marked expected-to-fail) ---
{tests}

Produce a Markdown report with these sections:

# Weaknesses Report — {file}

## Summary
A 2-3 sentence overview of the code's overall robustness.

## Findings
A numbered list. For EACH finding provide:
- **Weakness**: what is wrong or uncovered (name the specific input/scenario)
- **Impact**: what breaks, and how bad it is (low / medium / high)
- **Evidence**: the function/line or the test that exposes it
- **Suggested fix**: a concrete change, with a short code snippet if helpful

## Uncovered cases
A bullet list of scenarios the current tests/code do NOT handle but should.

## Rule compliance
For each business rule, state whether the code appears to ENFORCE it, and if not,
what to add.

Be concrete and reference actual identifiers from the source. Output only the
Markdown report."""


def run_analysis(client, target_desc, rel_file, source, rules_block, test_code):
    """Second LLM pass: produce a Markdown weaknesses report with fixes."""
    prompt = ANALYSIS_PROMPT.format(
        target_desc=target_desc, file=rel_file,
        source=source, rules=rules_block, tests=test_code,
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return strip_fences(resp.choices[0].message.content)


# ── Main ──────────────────────────────────────────────────────────────────────

def find_top_file_for_prompt(prompt, repo_name, min_score=0.3):
    """Uses the shared GroundworkRetriever (file-level) to find the single most
    associated file for a prompt. Returns (rel_file, score) or (None, None)."""
    retriever = GroundworkRetriever(top_n=1, repo_name=repo_name,
                                    mode="file", min_score=min_score)
    docs = retriever.invoke(prompt)
    if not docs:
        return None, None
    top = docs[0]
    return top.metadata.get("relative"), top.metadata.get("score")


IMPORT_PATTERNS = {
    ".py": [
        # from module import a, b, c
        (r'^\s*from\s+([\w.]+)\s+import\s+(.+)$', "from"),
        # import module
        (r'^\s*import\s+([\w.]+)', "plain"),
    ],
}


def find_imported_functions(source, rel_file, repo_name):
    """
    Parses the target file's imports and resolves them against the repo's
    dependency files (from Kùzu) to find imported FUNCTIONS worth testing.

    Returns a list of (dep_file, function_name) pairs. Only functions imported
    from files that are actually in this repo's knowledge base are returned.
    """
    ext = Path(rel_file).suffix.lower()
    patterns = IMPORT_PATTERNS.get(ext)
    if not patterns:
        return []

    # Dependency files this file imports from (repo-scoped, from Kùzu)
    deps_map = kb_get_dependencies([rel_file], repo_name)
    dep_files = deps_map.get(rel_file, [])
    # Index dep files by module stem for resolution: "helpers" -> "flask/helpers.py"
    by_stem = {}
    for d in dep_files:
        by_stem[Path(d).stem] = d

    import re
    found = []
    for line in source.splitlines():
        for pat, kind in patterns:
            m = re.match(pat, line)
            if not m:
                continue
            if kind == "from":
                module = m.group(1)              # e.g. "flask.helpers" or "helpers"
                names = m.group(2)
                mod_stem = module.split(".")[-1]  # last segment
                dep_file = by_stem.get(mod_stem)
                if not dep_file:
                    continue
                # Split imported names: "a, b as c, d"
                for raw in names.split(","):
                    name = raw.strip().split(" as ")[0].strip().strip("()")
                    # Heuristic: function-like names (skip * and CONSTANTS/Classes optional)
                    if name and name != "*":
                        found.append((dep_file, name))
    # De-dupe
    seen, result = set(), []
    for pair in found:
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def resolve_repo(conn, requested):
    repos = list_repositories(conn)
    if not repos:
        sys.exit("Error: no repositories in the knowledge base. Ingest one first.")
    if requested:
        if requested not in repos:
            sys.exit(f"Error: repo '{requested}' not found. Available: {', '.join(repos)}")
        return requested
    if len(repos) == 1:
        return repos[0]
    sys.exit("Multiple repositories exist — specify one with --repo: " + ", ".join(repos))


def generate_for_target(client, repo_name, repo_path, rel_file, rules_by_file,
                        function=None, related=3, analyze=True,
                        to_stdout=False, out_dir="generated_tests",
                        reports_dir="weakness", report_format="pdf",
                        example_file_override=None):
    """
    Generates tests (and optionally a weaknesses report) for one target —
    either a whole file or a single function within it. Reusable so the same
    logic runs for the top file AND for each imported function.

    example_file_override, when given, is a path to an existing test file
    the user has manually pointed at (CLI --example-file, or the TUI's Tests
    tab "Example test file" field) — it's used INSTEAD OF the automatic
    search in detect_framework_from_existing_tests, for cases where the
    right example isn't findable by convention (or the user just wants to
    be explicit about it).
    """
    full_source = read_source(repo_path, rel_file)
    if function:
        source = extract_function(full_source, function, Path(rel_file).suffix.lower())
        target_desc = f"function '{function}' in {rel_file}"
    else:
        source = full_source
        target_desc = f"file {rel_file}"

    framework, _ = infer_framework(rel_file)
    if example_file_override:
        manual_framework, example_file, example_source = load_manual_example(example_file_override)
        if manual_framework:
            framework = manual_framework
            framework_source = f"manually attached: {example_file}"
        elif example_source:
            framework_source = (f"extension guess (attached {example_file} didn't match a known "
                               f"framework signature, but is still used as a style example)")
        else:
            framework_source = f"extension guess (attached file unreadable: {example_file})"
    else:
        detected_framework, example_file, example_source = detect_framework_from_existing_tests(
            repo_name, repo_path, rel_file=rel_file)
        framework_source = "extension guess"
        if detected_framework:
            framework = detected_framework
            framework_source = f"detected from {example_file}"
    file_rules = rules_by_file.get(rel_file, [])
    rules_block = "\n".join(f"- {r}" for r in file_rules) if file_rules else "(none recorded)"
    # Always ground signatures from the FULL file, not the (possibly
    # function-truncated) `source` above — a single targeted method still
    # needs its class's constructor-injected collaborators mocked correctly.
    ctx = gather_related_context(repo_name, rel_file, full_source, repo_path=repo_path, top_n=related)

    print(f"\n  ── Target: {target_desc} ──")
    print(f"     Framework   : {framework} ({framework_source})")
    print(f"     File rules  : {len(file_rules)}")
    print(f"     Dependencies: {ctx['dep_count']}")
    print(f"     Generating tests...")

    if example_source:
        # A style example, not full grounding — cap it the same way
        # dependency sources are capped.
        capped = example_source if len(example_source) <= 6000 else \
            example_source[:6000] + f"\n// ... truncated, {len(example_source) - 6000} more characters"
        example_block = (f"// ── {example_file} (an existing test already in this repo — match "
                         f"its naming, structure, mocking style, and assertion style) ──\n{capped}")
    else:
        example_block = "(no existing test file found in this repo to use as a style example)"

    prompt = TEST_PROMPT.format(
        framework=framework, target_desc=target_desc,
        repo=repo_name, file=rel_file,
        source=source, rules=rules_block,
        deps=ctx["deps_block"], signatures=ctx["signatures_block"],
        dep_sources=ctx["dep_sources_block"],
        example=example_block,
        key_points=ctx["key_points_block"],
        related=ctx["related_block"],
    )
    system_message = TEST_SYSTEM.format(
        xfail_mechanism=XFAIL_MECHANISM.get(framework, DEFAULT_XFAIL_MECHANISM)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": prompt}],
    )
    test_code = strip_fences(resp.choices[0].message.content)

    report = None
    if analyze:
        print("     Analyzing for weaknesses...")
        report = run_analysis(client, target_desc, rel_file, source, rules_block, test_code)

    # Build output name
    out_name = output_filename(rel_file)
    if function:
        stem, suffix = Path(out_name).stem, Path(out_name).suffix
        out_name = f"{stem}_{function}{suffix}"

    if to_stdout:
        print("\n" + "=" * 70 + f"  TESTS — {target_desc}\n")
        print(test_code)
        if report:
            print("\n" + "=" * 70 + f"  WEAKNESSES — {target_desc}\n")
            print(report)
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / out_name).write_text(test_code, encoding="utf-8")
    print(f"     ✓ Wrote tests  → {out / out_name}")
    if report:
        # Weakness reports live in their own folder, separate from the test files,
        # so a test runner pointed at out_dir never tries to collect them.
        rdir = Path(reports_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        stem = Path(out_name).stem + "_weaknesses"

        if report_format in ("md", "both"):
            md_path = rdir / f"{stem}.md"
            md_path.write_text(report, encoding="utf-8")
            print(f"     ✓ Wrote report → {md_path}")

        if report_format in ("pdf", "both"):
            pdf_path = rdir / f"{stem}.pdf"
            try:
                markdown_to_pdf(
                    report, pdf_path,
                    title=f"Weaknesses Report",
                    subtitle=f"{target_desc}  ·  {repo_name}  ·  {timestamp()}",
                )
                print(f"     ✓ Wrote report → {pdf_path}")
            except Exception as e:
                # Never lose the analysis because rendering failed — but say
                # clearly WHY, so a missing dependency isn't mistaken for
                # "it still writes markdown".
                print(f"     ! PDF rendering failed: {type(e).__name__}: {e}")
                print(f"       (check: pip install reportlab, and that "
                      f"kb/report_pdf.py exists)")
                fallback = rdir / f"{stem}.md"
                fallback.write_text(report, encoding="utf-8")
                print(f"     ✓ Wrote markdown fallback → {fallback}")


def main():
    parser = argparse.ArgumentParser(description="Generate unit tests from the Groundwork KB")
    parser.add_argument("--repo", default=None, help="Repository name")
    parser.add_argument("--file", default=None,
                        help="Target file, repo-name-prefixed (e.g. flask/src/flask/app.py) — "
                             "matches what's stored in PostgreSQL/Kùzu")
    parser.add_argument("--prompt", default=None,
                        help="Instead of --file: find the file most associated with this prompt")
    parser.add_argument("--function", default=None, help="Optional: a single function/method to target")
    parser.add_argument("--example-file", default=None,
                        help="Path to an existing test file on disk to use as a style/framework "
                             "reference, overriding the automatic search for one")
    parser.add_argument("--follow-imports", action="store_true",
                        help="Also test functions this file imports from repo dependencies")
    parser.add_argument("--repo-path", default=None,
                        help="Path to the repo on disk (default: ./repos/<repo>)")
    parser.add_argument("--out", default="generated_tests",
                        help="Directory to write the test files (default: ./generated_tests)")
    parser.add_argument("--print", dest="to_stdout", action="store_true",
                        help="Print to stdout instead of writing files")
    parser.add_argument("--related", type=int, default=3,
                        help="How many related files to retrieve for context (default: 3)")
    parser.add_argument("--min-score", type=float, default=0.3,
                        help="Min retrieval score when using --prompt (default: 0.3)")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Skip the weaknesses report (generate tests only)")
    parser.add_argument("--reports-dir", default="weakness",
                        help="Folder for weakness reports (default: weakness)")
    parser.add_argument("--report-format", default="pdf",
                        choices=["pdf", "md", "both"],
                        help="Weakness report format (default: pdf)")
    args = parser.parse_args()

    if not args.file and not args.prompt:
        sys.exit("Error: pass either --file <path> or --prompt \"...\"")

    conn = get_connection()
    try:
        repo_name = resolve_repo(conn, args.repo)
        rules_by_file = load_business_rules_from_db(conn, repo_name)
    finally:
        conn.close()

    # Resolve repo path on disk
    repo_path = Path(args.repo_path) if args.repo_path else Path("./repos") / repo_name
    if not repo_path.is_dir():
        alt = Path(repo_name)
        repo_path = alt if alt.is_dir() else repo_path
    if not repo_path.is_dir():
        sys.exit(f"Error: repo path not found. Pass --repo-path. Tried: {repo_path}")

    # Resolve the target file — either given directly or found from the prompt
    if args.prompt:
        print(f"  Finding the file most associated with: \"{args.prompt}\"")
        rel_file, score = find_top_file_for_prompt(args.prompt, repo_name, args.min_score)
        if not rel_file:
            sys.exit("  No file passed the retrieval threshold. Try a different prompt "
                     "or lower --min-score.")
        print(f"  Top file: {rel_file}  (score {score:.3f})")
    else:
        rel_file = args.file

    client = get_client()

    # 1. Generate tests for the primary target
    generate_for_target(
        client, repo_name, repo_path, rel_file, rules_by_file,
        function=args.function, related=args.related,
        analyze=not args.no_analyze, to_stdout=args.to_stdout, out_dir=args.out,
        reports_dir=args.reports_dir, report_format=args.report_format,
        example_file_override=args.example_file,
    )

    # 2. Optionally follow imports: test functions imported from repo dependencies
    if args.follow_imports and not args.function:
        source = read_source(repo_path, rel_file)
        imported = find_imported_functions(source, rel_file, repo_name)
        if imported:
            print(f"\n  Following {len(imported)} imported function(s) from dependencies...")
            for dep_file, func_name in imported:
                try:
                    generate_for_target(
                        client, repo_name, repo_path, dep_file, rules_by_file,
                        function=func_name, related=args.related,
                        analyze=not args.no_analyze, to_stdout=args.to_stdout, out_dir=args.out,
                        reports_dir=args.reports_dir, report_format=args.report_format,
                        example_file_override=args.example_file,
                    )
                except SystemExit as e:
                    print(f"     (skipped {func_name} in {dep_file}: {e})")
        else:
            print("\n  No imported functions resolved to repo dependency files.")

    if not args.to_stdout:
        print(f"\n    Review before running — generated tests are a starting point, not ground truth.")


if __name__ == "__main__":
    main()