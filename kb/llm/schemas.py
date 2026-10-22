"""
Pydantic schemas for every structured LLM reply in Groundwork's ingestion
and test-generation graphs. (Translation's schemas live in
kb/translate/components.py.)

Rule for every class here: no default values and no dict[str, Any] fields,
because OpenAI's strict json_schema mode (LangChain's default for
with_structured_output) rejects both.
"""

from pydantic import BaseModel, Field


# ── Business rule extraction ──────────────────────────────────────────────────

class ExtractedRules(BaseModel):
    """Business rules for one file, each a user story."""
    rules: list[str] = Field(
        description='User stories: "As a [role], I want [action] so that [benefit]." '
                    "Empty list if the file has no business logic.")


class RuleCritique(BaseModel):
    """The interrogator's verdict on a set of extracted rules, checked
    against the source code. Drives the extract → interrogate → refine loop."""
    approved: bool = Field(description="True if the rules need no changes.")
    unsupported: list[str] = Field(
        description="Rules that claim behaviour the code does not implement, quoted exactly.")
    missing: list[str] = Field(
        description="Behaviours in the code (validation, branching, limits, error cases) "
                    "with no corresponding rule, each described in one sentence.")
    vague: list[str] = Field(
        description="Rules too vague to test, quoted exactly, each with what is unclear.")


# ── Key point synthesis ───────────────────────────────────────────────────────

class Capabilities(BaseModel):
    points: list[str] = Field(description="Distinct, specific capability statements.")


# ── Test generation ───────────────────────────────────────────────────────────

class GeneratedTest(BaseModel):
    code: str = Field(description="The complete pytest test file, without markdown fences.")
    covered: list[str] = Field(description="Behaviours or rules the tests cover, one per entry.")


class TestReflection(BaseModel):
    """Diagnosis of a failed test run, fed back to the generator."""
    __test__ = False                  # not a pytest class
    cause: str = Field(description="What went wrong, in one or two sentences.")
    fault: str = Field(description='"test" if the test is wrong, "code" if the test '
                                   'exposes a real bug in the source, "environment" for '
                                   "import/setup problems.")
    fixes: list[str] = Field(description="Concrete changes for the next attempt.")
