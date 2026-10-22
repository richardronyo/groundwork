"""
Groundwork — translation pipeline schemas (AgentModernize-style).

Shapes follow bsg_components.py from the standalone pipeline:
BusinessRule → RuleInventory → BSGNode/BSGEdge → BSG, plus the language-
neutral test cases and the translation output.

Every class is LLM-facing, so none has default values or free-form dict
fields (OpenAI strict json_schema rejects both). Test arguments therefore
travel as a JSON string (`args_json`) rather than a dict.
"""

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["HIGH", "MEDIUM", "LOW"]


class BusinessRule(BaseModel):
    description: str = Field(description="One rule, stated as observable behaviour.")
    category: Literal["EXPLICIT", "IMPLICIT"] = Field(
        description="EXPLICIT if written as a condition in code; IMPLICIT if it follows "
                    "from how the code behaves (defaults, skipped values, ordering).")
    confidence: Confidence
    filename: str


class RuleInventory(BaseModel):
    rules: list[BusinessRule]


class FieldSpecification(BaseModel):
    name: str
    type: str = Field(description="Language-neutral type: string, int, decimal, bool, date, "
                                  "list<T>, record{field:T}, optional<T>, enum[A,B].")


class BSGNode(BaseModel):
    name: str = Field(description="A behaviour, e.g. ValidateOrder — not necessarily a function.")
    functions: list[str] = Field(
        description="Source functions/methods (exact names) that implement this behaviour.")
    rules: list[BusinessRule]
    inputs: list[FieldSpecification]
    outputs: list[FieldSpecification]
    preconditions: list[str]
    postconditions: list[str]
    confidence: Confidence


class BSGEdge(BaseModel):
    source: str = Field(description="Node name, or '<file path>::<node>' for another file's node.")
    target: str = Field(description="Node name, or '<file path>::<node>' for another file's node.")
    label: Literal["sequence", "conditional", "data", "error", "depends_on"]


class BSG(BaseModel):
    nodes: list[BSGNode]
    edges: list[BSGEdge]


# ── Language-neutral tests (see kb/verify/harness.py for the execution model) ─

class TestStep(BaseModel):
    __test__ = False                  # not a pytest class
    id: str = Field(description="Unique within the case: s1, s2, ...")
    function: str = Field(description="Exact source function or class name, or "
                                      "'<earlier step id>.<method>' to call a method on an "
                                      "earlier step's result.")
    args_json: str = Field(description='JSON array of positional arguments, e.g. "[100, \\"gold\\"]". '
                                       'Use {"$ref": "s1"} or {"$ref": "s1.field"} to pass an '
                                       "earlier step's result.")


class TestCase(BaseModel):
    __test__ = False                  # not a pytest class
    id: str
    node: str = Field(description="The BSG node this case tests.")
    kind: Literal["unit", "behaviour"] = Field(
        description="unit: one step calling one function. behaviour: several steps exercising "
                    "the node's behaviour end to end.")
    description: str
    steps: list[TestStep]


class TestSuite(BaseModel):
    __test__ = False                  # not a pytest class
    cases: list[TestCase]


# ── Translation output ────────────────────────────────────────────────────────

class SourceFile(BaseModel):
    path: str = Field(description="File name with the target language's extension.")
    content: str = Field(description="Complete file contents, no markdown fences.")


class Translation(BaseModel):
    files: list[SourceFile] = Field(description="The main translated file first.")
    notes: str = Field(description="Anything that could not be translated one-to-one.")
