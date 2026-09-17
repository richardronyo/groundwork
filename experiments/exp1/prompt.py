"""
Prompt library for extracting business rules from a code file.

Part 1 - General Results: three ways of asking for business rules with no
role framing, to compare against each other.

Part 2 - Role prompts: the same "best" general prompt from Part 1, with a
role/perspective layered on top. After you've run Part 1 and picked a
winner, set BEST_DEFINITION below to match it.

All prompts return the same JSON shape so downstream parsing doesn't need
to change based on which variant produced it.
"""

BUSINESS_RULE_CATEGORIES = """\
Look for rules like these:
- A validation or constraint (required fields, allowed ranges, format checks)
- A default value and when it applies
- A threshold or limit that triggers different behavior once crossed
- An access control or permission check (who can do what, and when)
- An error or exception condition and what happens when it fires
- A state transition or lifecycle rule (what must happen before/after something else)
- A calculated or derived value and the logic behind it
- A contract with an external system, config value, or environment variable"""

WHAT_NOT_TO_INCLUDE = """\
Do NOT report:
- Plain restatements of syntax ("this is a for loop", "this function returns a string")
- Implementation details with no behavioral consequence (variable names, formatting, imports)
- Anything you're inferring beyond what the code actually shows"""

OUTPUT_INSTRUCTIONS = """\
Return your answer as a JSON array of objects, and nothing else (no prose, \
no markdown fences). Each object must have exactly these two keys:
  "file": the file path given to you, exactly as given
  "business_rule": the rule, as one sentence

If the file has no identifiable business rules, return an empty array: []"""

# Two ways of defining what a business rule is. Use the general one unless
# the prompt variant specifically calls for the COBREX definition.

DEFINITION_GENERAL = (
    "For this task, a business rule is a business requirement or "
    "constraint that the code enforces, not just a description of what "
    "the code does."
)

DEFINITION_COBREX = (
    'According to the COBREX paper, a business rule is defined as '
    '"a constraint at the program level that calculates a business '
    'result." It further explains that "the outcome of a business '
    'decision reflects in a single value or a set of related values. '
    'A business rule is a function that generates these values."'
)

USER_STORY_INSTRUCTIONS = """\
Write each business rule as a user story, in this form:
"As a <role>, I want <capability or constraint>, so that <reason or business outcome>."
Infer a plausible role (for example "administrator", "end user", or "system") \
when the code doesn't name one explicitly. Put the finished user story sentence \
in the "business_rule" field. Everything else about what counts as a business \
rule stays the same."""

ROLE_INTROS = {
    "business_analyst": (
        "You are a business analyst reviewing this code to document its "
        "business behavior for non-technical stakeholders. Focus on what "
        "the code means for the business, not how it's implemented."
    ),
    "code_reviewer": (
        "You are a code reviewer auditing this file for hidden business "
        "logic that isn't obvious at a glance. Focus on rules a reviewer "
        "could easily miss, or that deserve a comment or a test."
    ),
    "senior_developer": (
        "You are a senior developer handing this file off to a new team "
        "member. Focus on the rules they'd need to know so they don't "
        "accidentally break behavior when they touch this code."
    ),
}


def _build_system_prompt(definition: str, extra_instructions: str = "", role_intro: str = "") -> str:
    """Assembles a full system prompt from the shared building blocks."""
    intro = role_intro or (
        "You are a senior software analyst. You read source code and pull "
        "out its business rules: the constraints, decisions, and behaviors "
        "the code enforces, not just what the code technically does."
    )

    parts = [intro, definition, BUSINESS_RULE_CATEGORIES, WHAT_NOT_TO_INCLUDE]

    if extra_instructions:
        parts.append(extra_instructions)

    parts.append(
        "Write each rule as a short, plain-English sentence a "
        "non-programmer could understand. Name the specific function, "
        "class, or config key involved when it helps identify where the "
        "rule lives. Do not use em dashes."
    )
    parts.append(OUTPUT_INSTRUCTIONS)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Part 1: General Results
# ---------------------------------------------------------------------------

# 1. General Prompt: just ask for the business rules.
PROMPT_GENERAL = _build_system_prompt(DEFINITION_GENERAL)

# 2. General Prompt + User Story: same as above, formatted as user stories.
PROMPT_USER_STORY = _build_system_prompt(
    DEFINITION_GENERAL, extra_instructions=USER_STORY_INSTRUCTIONS
)

# 3. General Prompt + Definition: swaps in the COBREX definition.
PROMPT_DEFINITION = _build_system_prompt(DEFINITION_COBREX)


# ---------------------------------------------------------------------------
# Part 2: Using the best General Prompt
# ---------------------------------------------------------------------------

# After running Part 1's comparison, set this to whichever definition
# belonged to the winning prompt. Defaults to the general definition;
# swap to DEFINITION_COBREX if PROMPT_DEFINITION wins instead.
BEST_DEFINITION = DEFINITION_GENERAL

# 4.1 Best Prompt + Business Analyst role
PROMPT_ROLE_BUSINESS_ANALYST = _build_system_prompt(
    BEST_DEFINITION, role_intro=ROLE_INTROS["business_analyst"]
)

# 4.2 Best Prompt + Code Reviewer role
PROMPT_ROLE_CODE_REVIEWER = _build_system_prompt(
    BEST_DEFINITION, role_intro=ROLE_INTROS["code_reviewer"]
)

# 4.3 Best Prompt + Senior Developer role
PROMPT_ROLE_SENIOR_DEVELOPER = _build_system_prompt(
    BEST_DEFINITION, role_intro=ROLE_INTROS["senior_developer"]
)


# Convenience lookup so callers can select a variant by name.
PROMPTS = {
    "general": PROMPT_GENERAL,
    "user_story": PROMPT_USER_STORY,
    "definition": PROMPT_DEFINITION,
    "role_business_analyst": PROMPT_ROLE_BUSINESS_ANALYST,
    "role_code_reviewer": PROMPT_ROLE_CODE_REVIEWER,
    "role_senior_developer": PROMPT_ROLE_SENIOR_DEVELOPER,
}


# ---------------------------------------------------------------------------
# Message construction (provider-agnostic {role, content} shape)
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
File: {file_path}

```{language}
{code}
```

Extract the business rules from this file."""


def build_messages(system_prompt: str, file_path: str, code: str, language: str = "") -> list[dict]:
    """
    Builds a provider-agnostic messages list for a single file, given a
    chosen system prompt (one of the PROMPTS values above, or a custom one).
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                file_path=file_path, language=language, code=code
            ),
        },
    ]