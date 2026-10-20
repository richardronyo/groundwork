COBREX_BR = """
A business rule is a constraint at the program level that calculates a
business result.

Extract every business rule from the source code below. State each rule
as one plain sentence, using only what the code enforces.

Follow these rules when writing the list:
- Each rule describes exactly one feature: a single behaviour, setting,
  default value, condition, or limit.
- Each rule is exactly one sentence.
- If describing a behaviour would take more than one sentence, split it
  into several one-sentence rules instead.
- Several rules may describe the same function or setting, and may
  overlap; prefer several short overlapping rules over one long rule.
"""

USER_STORY_BR = """
Extract every business rule from the source code below. Write each rule
as a user story, using only what the code enforces:

As a <role>, I want <behaviour>, so that <outcome>.

Follow these rules when writing the list:
- Each user story describes exactly one feature: a single behaviour,
  setting, default value, condition, or limit.
- Each user story is exactly one sentence; do not join several
  behaviours with "and" inside one story.
- If a behaviour has several parts, write a separate user story for each
  part instead.
- Several user stories may describe the same function or setting, and
  may overlap; prefer several short overlapping stories over one long one.
"""

SCORE_PROMPT = """
Score these business rules against the source code, from 1 (poor) to 5 (excellent):
- clarity: are the rules reader-friendly and clearly expressed, with each
  rule written as one sentence?
- accuracy: do the rules state what the code actually does?
- coverage: do the rules cover the important behaviour in the code, with
  each distinct feature (behaviour, setting, default, condition, or limit)
  stated in its own rule?
- overall: how well do the rules represent the code as a whole, with each
  rule describing exactly one feature?
Lower the scores for rules that run to more than one sentence or combine
several features. Do not lower the scores because several rules overlap or
describe the same function; that is expected.
Then say in "feedback" what specific changes would raise the lowest scores.
"""

REFINER_PROMPT = """
Below are business rules extracted from the source code, with scores from
0 to 10 for their clarity, accuracy, coverage, and overall quality.

Improve the list to raise these scores, focusing on the lowest ones:
- Clarity: rewrite rules that are hard to read or ambiguous, and split any
  rule that is longer than one sentence into several one-sentence rules.
- Accuracy: correct rules that do not match what the code enforces.
- Coverage: add rules for important behaviour the list is missing, one
  rule per feature (behaviour, setting, default, condition, or limit).
- Overall quality: split rules that describe more than one feature,
  remove exact duplicates that state the same thing, and remove anything
  that is not a business rule.

Overlapping rules that each state a different part of the same function
are not duplicates; keep them separate and do not merge them.

Keep rules that are already correct unchanged, and keep every rule in the
same format as the existing ones. Use only what the code enforces.
Return the complete updated list.
"""

SIMILARITY_PROMPT = '''Rate how similar the meaning of two statements is, as a value between 0 and 1,
where 0 means completely different and 1 means identical in meaning.
Judge meaning, not wording. Each statement is a business rule describing one feature,
which may be written as a plain sentence or as a user story; ignore that difference in format.
If both describe the same feature but one adds or omits a condition, default, or value,
the meaning is not identical.
Give a one-sentence reason, then the score.'''

SIMILARITY_REVIEW_PROMPT = '''You scored this pair before. Check the previous score against the two statements.
Make sure the score judged the meaning of the feature each statement describes, not its
wording or format (plain sentence or user story), and that any added or missing condition,
default, or value was taken into account.
Keep it if it is right, or give a corrected score if something was missed.'''