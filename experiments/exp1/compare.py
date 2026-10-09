import pandas as pd

from pathlib import Path
from bert_score import score as bert_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from evaluators import compute_meteor, compute_rouge, CODE_DOCUMENTATION_MAP
from similarity import sim_app, MAX_SIM_ITERATION

GROUND_TRUTH = 'data/groundtruth.csv'
OUTPUT_FOLDER = 'outputs'
PROMPT_FORMATS = ['cobrex', 'user_story']
MAX_BR_ITERATION = 3

def best_matches(truths: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    #For each ground truth, finds the business rule with the highest cosine similarity,
    #looking only at rules extracted from the code files that correspond to its .rst file
    model = SentenceTransformer('all-MiniLM-L6-v2')
    code_file = rules['filename'].map(lambda name: Path(name).name)     #json/provider.py -> provider.py
    rows = []

    for doc, group in truths.groupby('file', sort = False):
        code_files = [code for code, docs in CODE_DOCUMENTATION_MAP.items() if doc in docs]
        candidates = rules[code_file.isin(code_files)]

        if candidates.empty:
            #No rules were extracted for this document's code files, so the ground truths stay unmatched
            rows += [{'file': doc, 'ground_truth': truth} for truth in group['business_rule']]
            continue

        scores = cosine_similarity(model.encode(group['business_rule'].tolist()),
                                   model.encode(candidates['description'].tolist()))

        for truth, row in zip(group['business_rule'], scores):
            best = candidates.iloc[row.argmax()]
            rows.append({'file': doc, 'ground_truth': truth, 'business_rule': best['description'],
                         'rule_file': best['filename'], 'cosine': row.max()})

    return pd.DataFrame(rows, columns = ['file', 'ground_truth', 'business_rule', 'rule_file', 'cosine'])

def score_matches(matches: pd.DataFrame) -> pd.DataFrame:
    #Adds the METEOR, ROUGE, BERT and LLM scores for every ground truth that has a matched business rule
    matched = matches.dropna(subset = ['business_rule'])
    if matched.empty:
        return matches

    references, hypotheses = matched['ground_truth'].tolist(), matched['business_rule'].tolist()
    scores = pd.DataFrame(index = matched.index)

    scores['meteor'] = [compute_meteor(r, h) for r, h in zip(references, hypotheses)]

    rouge = [compute_rouge(r, h) for r, h in zip(references, hypotheses)]
    for name in ('rouge1', 'rouge2', 'rougeL'):
        for part in ('precision', 'recall', 'fmeasure'):
            scores[f'{name}_{part}'] = [result[name][part] for result in rouge]

    #Same call as compute_bert, but on every pair at once so the BERT model only loads one time
    P, R, F1 = bert_scorer(hypotheses, references, lang = 'en', verbose = False)
    scores['bert_precision'], scores['bert_recall'], scores['bert_f1'] = P.tolist(), R.tolist(), F1.tolist()

    #LLM evaluator: one score per iteration, repeating the last one if the loop stopped early
    results = sim_app.batch([{'statement_a': r, 'statement_b': h} for r, h in zip(references, hypotheses)])
    passes = [[entry['score'] for entry in result['history']] for result in results]
    passes = [p + [p[-1]] * (MAX_SIM_ITERATION - len(p)) for p in passes]
    for number in range(MAX_SIM_ITERATION):
        scores[f'llm_score_{number + 1}'] = [p[number] for p in passes]
    scores['llm_score'] = [result['score'] for result in results]
    scores['llm_reason'] = [result['reason'] for result in results]

    return matches.join(scores)

def compare(rules_path: str, output_path: str, truth_path: str = GROUND_TRUTH) -> pd.DataFrame:
    #Compares one file of extracted business rules to the ground truths and saves the result
    truths, rules = pd.read_csv(truth_path), pd.read_csv(rules_path)
    result = score_matches(best_matches(truths, rules))
    result.to_csv(output_path, index = False)
    return result

if __name__ == '__main__':
    for prompt_format in PROMPT_FORMATS:
        for iteration in range(MAX_BR_ITERATION + 1):
            rules_path = f'{OUTPUT_FOLDER}/{prompt_format}/br{iteration}.csv'
            if not Path(rules_path).exists():
                print(f'{rules_path} not found, skipped')
                continue

            result = compare(rules_path, f'{OUTPUT_FOLDER}/{prompt_format}/comparison{iteration}.csv')
            unmatched = result['business_rule'].isna().sum()
            print(f'{prompt_format} iteration {iteration}: {len(result)} ground truths, {unmatched} with no rules to compare against')