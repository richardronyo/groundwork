COBREX_BR = """
A business rule is a constraint at the program level that calculates a
business result.

Extract every business rule from the source code below. State each rule
as one plain sentence, using only what the code enforces.
"""

USER_STORY_BR = """
Extract every business rule from the source code below. Write each rule
as a user story, using only what the code enforces:

As a <role>, I want <behaviour>, so that <outcome>.
"""

SCORE_PROMPT = """
Score these business rules against the source code, from 1 (poor) to 5 (excellent):
- clarity: are the rules reader-friendly and clearly expressed?
- accuracy: do the rules state what the code actually does?
- coverage: do the rules cover the important behaviour in the code?
- overall: how well do the rules represent the code as a whole?
Then say in "feedback" what specific changes would raise the lowest scores.
"""

REFINER_PROMPT = """
Below are business rules extracted from the source code, with scores from
0 to 10 for their clarity, accuracy, coverage, and overall quality.

Improve the list to raise these scores, focusing on the lowest ones:
- Clarity: rewrite rules that are hard to read or ambiguous.
- Accuracy: correct rules that do not match what the code enforces.
- Coverage: add rules for important behaviour the list is missing.
- Overall quality: remove duplicates and anything that is not a business rule.

Keep rules that are already correct unchanged, and keep every rule in the
same format as the existing ones. Use only what the code enforces.
Return the complete updated list.
"""

SIMILARITY_PROMPT = '''Rate how similar the meaning of two statements is, as a value between 0 and 1,
where 0 means completely different and 1 means identical in meaning.
Judge meaning, not wording. Give a one-sentence reason, then the score.'''

SIMILARITY_REVIEW_PROMPT = '''You scored this pair before. Check the previous score against the two statements.
Keep it if it is right, or give a corrected score if something was missed.'''