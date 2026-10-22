"""
Groundwork — language-agnostic test execution for the translation graph.

A test case is data, not code, so the same case runs against the source file
and its translation:

    {"id": "t3", "node": "CalculateDiscount", "kind": "behaviour",
     "steps": [
        {"id": "s1", "function": "make_order", "args_json": "[[10.0, 5.0], [1, 2]]"},
        {"id": "s2", "function": "apply_discount", "args_json": "[{\"$ref\": \"s1\"}, \"gold\"]"},
        {"id": "s3", "function": "s2.total", "args_json": "[]"}       # method on an earlier result
     ]}

  - args_json is a JSON array of POSITIONAL arguments (translations keep
    parameter order, and not every language has keyword arguments).
  - {"$ref": "s1"} or {"$ref": "s1.field"} substitutes an earlier step's
    result (or a field of it).
  - "s2.total" as a function name calls method `total` on step s2's result.

Each language has a small harness script that loads the module, runs every
step and prints one JSON document:

    {"results": [{"id": "t3", "steps": [
        {"id": "s1", "ok": true, "value": ...},
        {"id": "s2", "ok": false, "error": "ValueError: ..."},     # code raised: a valid outcome
        {"id": "s3", "harness_error": "no function 'totl'"}]}]}    # the TEST is broken

A harness_error means the test itself is invalid (unknown function, bad JSON,
bad $ref) — the translation graph regenerates those. An exception from the
code is a legitimate expected outcome that the translation must reproduce.

Runners: python, javascript (node), r (Rscript + jsonlite). Adding a language
means adding one Runner with a harness script and an `export_note` that tells
the translator how to expose functions so the harness can find them.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TIMEOUT = 60

# ── Harness scripts ───────────────────────────────────────────────────────────

PY_HARNESS = r'''
import dataclasses, decimal, enum, importlib, importlib.util, json, math, sys

class HarnessError(Exception):
    pass

def to_json(v, depth=0):
    if depth > 20: return str(v)
    if v is None or isinstance(v, (bool, int, str)): return v
    if isinstance(v, float): return v if math.isfinite(v) else str(v)
    if isinstance(v, decimal.Decimal): return str(v)
    if isinstance(v, enum.Enum): return to_json(v.value, depth + 1)
    if isinstance(v, dict): return {str(k): to_json(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [to_json(x, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)): return sorted((to_json(x, depth + 1) for x in v), key=repr)
    if dataclasses.is_dataclass(v): return to_json(dataclasses.asdict(v), depth + 1)
    if hasattr(v, "__dict__"):
        return {k: to_json(x, depth + 1) for k, x in vars(v).items() if not k.startswith("_")}
    return str(v)

def load(module):
    if module.get("name"):
        return importlib.import_module(module["name"])
    spec = importlib.util.spec_from_file_location("gw_target", module["path"])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gw_target"] = mod
    spec.loader.exec_module(mod)
    return mod

def deref(ref, env):
    head, _, field = ref.partition(".")
    if head not in env: raise HarnessError(f"$ref to unknown or failed step '{head}'")
    v = env[head]
    for part in [p for p in field.split(".") if p]:
        v = v[part] if isinstance(v, dict) else getattr(v, part)
    return v

def subst(x, env):
    if isinstance(x, dict):
        if set(x) == {"$ref"}: return deref(x["$ref"], env)
        return {k: subst(v, env) for k, v in x.items()}
    if isinstance(x, list): return [subst(v, env) for v in x]
    return x

def resolve(name, mod, env):
    head, _, rest = name.partition(".")
    if rest and head in env:
        obj = env[head]
        for part in rest.split(".")[:-1]: obj = getattr(obj, part)
        fn = getattr(obj, rest.split(".")[-1], None)
    else:
        fn = mod
        for part in name.split("."):
            fn = getattr(fn, part, None)
            if fn is None: break
    if not callable(fn): raise HarnessError(f"no function '{name}'")
    return fn

def main():
    spec = json.load(open(sys.argv[1]))
    try:
        mod = load(spec["module"])
    except Exception as e:
        print(json.dumps({"load_error": f"{type(e).__name__}: {e}"})); return
    results = []
    for case in spec["cases"]:
        env, out = {}, []
        for step in case["steps"]:
            try:
                fn = resolve(step["function"], mod, env)
                args = json.loads(step["args_json"] or "[]")
                if not isinstance(args, list): raise HarnessError("args_json must be a JSON array")
                args = subst(args, env)
            except (HarnessError, ValueError, KeyError, AttributeError, TypeError) as e:
                out.append({"id": step["id"], "harness_error": f"{type(e).__name__}: {e}"}); break
            try:
                v = fn(*args)
                env[step["id"]] = v
                out.append({"id": step["id"], "ok": True, "value": to_json(v)})
            except Exception as e:
                out.append({"id": step["id"], "ok": False, "error": f"{type(e).__name__}: {e}"})
        results.append({"id": case["id"], "steps": out})
    print(json.dumps({"results": results}))

main()
'''

JS_HARNESS = r'''
const fs = require("fs");
const { pathToFileURL } = require("url");

class HarnessError extends Error {}

function toJSON(v, depth = 0) {
  if (depth > 20) return String(v);
  if (v === undefined || v === null) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : String(v);
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "boolean" || typeof v === "string") return v;
  if (v instanceof Map) return Object.fromEntries([...v].map(([k, x]) => [String(k), toJSON(x, depth + 1)]));
  if (v instanceof Set) return [...v].map(x => toJSON(x, depth + 1));
  if (Array.isArray(v)) return v.map(x => toJSON(x, depth + 1));
  if (typeof v === "object") {
    if (typeof v.toJSON === "function") return toJSON(v.toJSON(), depth + 1);
    const o = {};
    for (const k of Object.keys(v)) if (!k.startsWith("_")) o[k] = toJSON(v[k], depth + 1);
    return o;
  }
  return String(v);
}

function deref(ref, env) {
  const [head, ...rest] = ref.split(".");
  if (!(head in env)) throw new HarnessError(`$ref to unknown or failed step '${head}'`);
  let v = env[head];
  for (const p of rest) v = v[p];
  return v;
}

function subst(x, env) {
  if (Array.isArray(x)) return x.map(v => subst(v, env));
  if (x && typeof x === "object") {
    const keys = Object.keys(x);
    if (keys.length === 1 && keys[0] === "$ref") return deref(x["$ref"], env);
    const o = {}; for (const k of keys) o[k] = subst(x[k], env); return o;
  }
  return x;
}

const isClass = f => typeof f === "function" && /^class[\s{]/.test(Function.prototype.toString.call(f));

function resolve(name, ns, env) {
  const parts = name.split(".");
  if (parts.length > 1 && parts[0] in env) {
    let obj = env[parts[0]];
    for (const p of parts.slice(1, -1)) obj = obj[p];
    const fn = obj == null ? undefined : obj[parts[parts.length - 1]];
    if (typeof fn !== "function") throw new HarnessError(`no method '${name}'`);
    return (...a) => fn.apply(obj, a);
  }
  let fn = ns;
  for (const p of parts) { fn = fn == null ? undefined : fn[p]; }
  if (typeof fn !== "function") throw new HarnessError(`no function '${name}'`);
  return isClass(fn) ? (...a) => new fn(...a) : fn;
}

(async () => {
  const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  let mod;
  try { mod = await import(pathToFileURL(spec.module.path).href); }
  catch (e) { console.log(JSON.stringify({ load_error: `${e.name}: ${e.message}` })); return; }
  const ns = Object.assign({}, (mod.default && typeof mod.default === "object") ? mod.default : {}, mod);
  const results = [];
  for (const c of spec.cases) {
    const env = {}, out = [];
    for (const step of c.steps) {
      let fn, args;
      try {
        fn = resolve(step.function, ns, env);
        args = JSON.parse(step.args_json || "[]");
        if (!Array.isArray(args)) throw new HarnessError("args_json must be a JSON array");
        args = subst(args, env);
      } catch (e) { out.push({ id: step.id, harness_error: `${e.name}: ${e.message}` }); break; }
      try {
        let v = fn(...args);
        if (v && typeof v.then === "function") v = await v;
        env[step.id] = v;
        out.push({ id: step.id, ok: true, value: toJSON(v) });
      } catch (e) {
        out.push({ id: step.id, ok: false, error: `${e && e.name}: ${e && e.message}` });
      }
    }
    results.push({ id: c.id, steps: out });
  }
  console.log(JSON.stringify({ results }));
})();
'''

R_HARNESS = r'''
suppressMessages(library(jsonlite))
argv <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(argv[1], simplifyVector = FALSE)
emit <- function(x) cat(toJSON(x, auto_unbox = TRUE, null = "null", na = "null", digits = NA))
env <- new.env()
ok <- tryCatch({ sys.source(spec$module$path, envir = env); TRUE },
               error = function(e) { emit(list(load_error = conditionMessage(e))); FALSE })
if (ok) {
  deref <- function(ref, steps) {
    parts <- strsplit(ref, ".", fixed = TRUE)[[1]]
    if (is.null(steps[[parts[1]]])) stop(paste0("HARNESS: $ref to unknown or failed step '", parts[1], "'"))
    v <- steps[[parts[1]]]
    for (p in parts[-1]) v <- v[[p]]
    v
  }
  subst <- function(x, steps) {
    if (is.list(x)) {
      if (!is.null(names(x)) && identical(names(x), "$ref")) return(deref(x[["$ref"]], steps))
      return(lapply(x, subst, steps = steps))
    }
    x
  }
  results <- list()
  for (cs in spec$cases) {
    steps <- list(); out <- list()
    for (st in cs$steps) {
      prep <- tryCatch({
        fn <- get0(st$`function`, envir = env, mode = "function", inherits = FALSE)
        if (is.null(fn)) stop(paste0("HARNESS: no function '", st$`function`, "'"))
        a <- fromJSON(if (is.null(st$args_json) || st$args_json == "") "[]" else st$args_json,
                      simplifyVector = TRUE)
        a <- if (is.list(a)) a else as.list(a)
        list(fn = fn, args = subst(a, steps))
      }, error = function(e) list(err = conditionMessage(e)))
      if (!is.null(prep$err)) { out[[length(out) + 1]] <- list(id = st$id, harness_error = prep$err); break }
      res <- tryCatch(list(ok = TRUE, value = do.call(prep$fn, unname(prep$args))),
                      error = function(e) list(ok = FALSE, error = conditionMessage(e)))
      if (isTRUE(res$ok)) steps[[st$id]] <- res$value
      out[[length(out) + 1]] <- c(list(id = st$id), res)
    }
    results[[length(results) + 1]] <- list(id = cs$id, steps = out)
  }
  emit(list(results = results))
}
'''


# ── Runners ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Runner:
    key: str
    extension: str
    command: tuple[str, ...]          # interpreter + harness path is appended
    harness: str
    harness_ext: str
    export_note: str                  # told to the translator

    def available(self) -> bool:
        return shutil.which(self.command[0]) is not None


RUNNERS: dict[str, Runner] = {
    "python": Runner(
        "python", ".py", (sys.executable,), PY_HARNESS, ".py",
        "Define every function and class at module top level with the same names as the "
        "source. The test harness imports the module and calls them by name with "
        "positional arguments."),
    "javascript": Runner(
        "javascript", ".js", ("node",), JS_HARNESS, ".js",
        "Write CommonJS for Node.js. End the file with `module.exports = { ... }` listing "
        "every function and class, under the same names as the source. The test harness "
        "requires the module and calls them by name with positional arguments; Python "
        "exceptions become thrown Errors."),
    "r": Runner(
        "r", ".R", ("Rscript",), R_HARNESS, ".R",
        "Define every function at top level with the same names as the source. The test "
        "harness sources the file and calls functions by name with positional arguments "
        "via do.call(); raise errors with stop()."),
}

# tree_to_json language names → runner keys
LANGUAGE_TO_RUNNER = {"Python": "python", "JavaScript": "javascript", "R": "r"}


def runner_for(language: str) -> Runner | None:
    key = LANGUAGE_TO_RUNNER.get(language, language.lower() if language else "")
    r = RUNNERS.get(key)
    return r if r and r.available() else None


def _pythonpath(repo_root: Path | None, module_dir: Path | None = None) -> str:
    parts = [str(module_dir)] if module_dir else []
    if repo_root:
        parts += [str(repo_root), str(repo_root / "src"), str(repo_root.parent)]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)


def python_module_name(repo_root: Path, file_path: Path) -> str | None:
    """Dotted import name for a file inside a package, so relative imports
    work when the SOURCE is executed (e.g. src/flask/helpers.py → flask.helpers).
    None for a loose script, which is then loaded by path instead."""
    file_path = file_path.resolve()
    for base in (repo_root / "src", repo_root):
        try:
            rel = file_path.relative_to(base.resolve())
        except ValueError:
            continue
        if (file_path.parent / "__init__.py").exists():
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
        if len(rel.parts) == 1:
            return None
    return None


def run_cases(runner: Runner, module_path: Path, cases: list[dict],
              repo_root: Path | None = None, module_name: str | None = None,
              timeout: int = TIMEOUT) -> dict:
    """Runs `cases` against one module. Returns the harness JSON
    ({"results": [...]} or {"load_error": ...}); never raises."""
    payload = {"module": {"path": str(Path(module_path).resolve()), "name": module_name},
               "cases": cases}
    with tempfile.TemporaryDirectory(prefix="gw_harness_") as tmp:
        spec_path = Path(tmp) / "spec.json"
        spec_path.write_text(json.dumps(payload))
        harness_path = Path(tmp) / f"harness{runner.harness_ext}"
        harness_path.write_text(runner.harness)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": _pythonpath(repo_root, Path(module_path).resolve().parent)}
        try:
            proc = subprocess.run([*runner.command, str(harness_path), str(spec_path)],
                                  capture_output=True, text=True, timeout=timeout,
                                  cwd=str(Path(module_path).resolve().parent), env=env)
        except subprocess.TimeoutExpired:
            return {"load_error": f"timeout after {timeout}s"}
    lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
    if not lines:
        return {"load_error": (proc.stderr or proc.stdout or "no output").strip()[-2000:]}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"load_error": proc.stdout[-2000:]}


# ── Comparing outcomes across languages ───────────────────────────────────────

def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def values_equal(a, b, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Structural equality that forgives cross-language representation:
    1 == 1.0 == "1.00" (decimals travel as strings), a 1-element list equals
    its element (R has no scalars), and object vs dict compare by fields."""
    if a == b:
        return True
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, list) and len(a) == 1 and not isinstance(b, list):
        return values_equal(a[0], b)
    if isinstance(b, list) and len(b) == 1 and not isinstance(a, list):
        return values_equal(a, b[0])
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if a in (None, [], {}) and b in (None, [], {}):
        return True
    return False


def compare_case(expected: dict, actual: dict | None) -> tuple[bool, str]:
    """Compares one case's step outcomes. Errors match if both sides errored
    (messages differ across languages, so only the fact of failure counts)."""
    if actual is None:
        return False, "case did not run"
    by_id = {s["id"]: s for s in actual["steps"]}
    for exp in expected["steps"]:
        act = by_id.get(exp["id"])
        if act is None:
            return False, f"step {exp['id']} did not run"
        if "harness_error" in act:
            return False, f"step {exp['id']}: {act['harness_error']}"
        if exp["ok"] != act.get("ok"):
            if exp["ok"]:
                return False, f"step {exp['id']}: expected {json.dumps(exp['value'])[:200]}, got error {act.get('error')}"
            return False, f"step {exp['id']}: expected an error ({exp['error']}), got {json.dumps(act.get('value'))[:200]}"
        if exp["ok"] and not values_equal(exp["value"], act.get("value")):
            return False, (f"step {exp['id']}: expected {json.dumps(exp['value'])[:200]}, "
                           f"got {json.dumps(act.get('value'))[:200]}")
    return True, ""
