"""Prompts for the translation graph (analyzer, specifier, test generator, translator)."""

ANALYZER_SYSTEM = """You are a business-rule analyst working on legacy code modernization.
You read source code and produce a complete inventory of the business rules it implements:
validations, calculations, thresholds, branching decisions, error conditions, defaults and
ordering effects. A rule must be supported by the code you are shown."""

ANALYZER_PROMPT = """Source file `{filename}` ({language}):

```
{code}
```
{kb_section}
Produce the rule inventory for this file. Mark each rule EXPLICIT (written as a condition)
or IMPLICIT (follows from behaviour, e.g. skipped missing values), with your confidence.
Use "{filename}" as each rule's filename."""

KB_RULES_SECTION = """
Groundwork already extracted these rules for this file (as user stories). Verify each
against the code: keep the supported ones (restated as behaviour), drop unsupported
ones, and add anything they miss — implicit rules especially:
{rules}
"""

SPECIFIER_SYSTEM = """You build Behavioural Specification Graphs (BSGs) from business rules.
A BSG node is a behaviour (ValidateOrder, CalculateDiscount), not necessarily one function.
Each node lists the exact source functions that implement it, its rules, language-neutral
inputs/outputs, and pre/postconditions precise enough to write tests from.
Edges record how behaviours connect: sequence, conditional, data, error, or depends_on
for behaviour provided by another file."""

SPECIFIER_PROMPT = """Source file `{filename}` ({language}):

```
{code}
```

Rule inventory:
{rules}
{structure_section}{deps_section}
Build the BSG for this file. Every rule must belong to exactly one node."""

STRUCTURE_SECTION = """
Functions and classes in this file (from Groundwork's parser):
{structure}
"""

DEPS_SECTION = """
This file depends on other files in the repository. Their behaviours are below; when a node
here relies on one, add an edge labelled depends_on whose target is '<file path>::<node name>'.
{deps}
"""

TESTGEN_SYSTEM = """You write language-neutral test cases for one behaviour of a program.
Each case is a list of steps; each step calls one function of the source module by exact
name with positional arguments given as a JSON array. You do NOT write expected values:
the cases are run against the original code to record what it actually does.
Pick inputs that exercise every rule, boundary and error path of the behaviour."""

TESTGEN_PROMPT = """Source file `{filename}` ({language}):

```
{code}
```

Behaviour under test:
{node}

Write {n_unit} or more unit cases (one step each) and at least one behaviour case
(several steps, later steps may pass {{"$ref": "s1"}} to use an earlier step's result,
or call "s1.method" on it). Prefix every case id with "{prefix}".
Only use functions and classes that exist in the source, with argument types they accept.
{feedback}"""

TESTGEN_FEEDBACK = """
Your previous cases could not be run against the source:
{problems}
Fix them (wrong function names, wrong number of arguments, invalid JSON, bad $ref)."""

TRANSLATOR_SYSTEM = """You translate source code from {source_language} to {target_language}.
Translate line by line: keep the same function and class names, the same parameter order,
the same control flow and the same comments, so a reader can compare the two side by side.
Use the BSG as a checklist that every behaviour survives — not as a blueprint to reorganise
the code around. Reproduce edge-case behaviour exactly, including error conditions,
rounding and handling of missing values. Use only the standard library unless the source
relies on a package with an obvious {target_language} equivalent.

{export_note}"""

TRANSLATOR_PROMPT = """Source file `{filename}` ({source_language}):

```
{code}
```

Behavioural Specification Graph (checklist):
{bsg}
{feedback}
Return the complete {target_language} translation; the main file should be named `{target_name}`."""

TRANSLATOR_FEEDBACK = """
Round {round} of verification ran the test cases against both the original and your
translation. {n_fail} of {n_total} cases disagree with the original:
{failures}

The best translation so far (round {best_round}, {best_rate:.0%} passing) is:

```
{best_code}
```

Fix the disagreements starting from that version. Do not break cases that already pass."""
