import pandas as pd

from pathlib import Path
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from evaluators import compute_meteor, compute_rouge, CODE_DOCUMENTATION_MAP
from similarity import sim_app, MAX_SIM_ITERATION

GROUND_TRUTH = 'data/groundtruth2.csv'
OUTPUT_FOLDER = 'outputs'
PROMPT_FORMATS = ['cobrex']
MAX_BR_ITERATION = 3
LLM_CONCURRENCY = 8     #How many LLM similarity requests run at the same time; raise it if your rate limit allows
TOP_K = 3               #Only each ground truth's TOP_K highest-cosine rules get the BERT and LLM scores

#Loaded once and shared by every compare() call instead of reloading for each file
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
BERT = BERTScorer(lang = 'en')

#(ground truth, business rule) -> LLM result, so a pair that shows up again in a later file is not re-sent
LLM_CACHE = {}

def all_pairs(truths: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    #Pairs every ground truth with every business rule extracted from the code files
    #that correspond to its .rst file, and gives each pair its cosine similarity
    code_file = rules['filename'].map(lambda name: Path(name).name)     #json/provider.py -> provider.py
    rows = []

    for doc, group in truths.groupby('file', sort = False):
        code_files = [code for code, docs in CODE_DOCUMENTATION_MAP.items() if doc in docs]
        candidates = rules[code_file.isin(code_files)]

        if candidates.empty:
            #No rules were extracted for this document's code files, so the ground truths stay unpaired
            rows += [{'file': doc, 'ground_truth': truth} for truth in group['business_rule']]
            continue

        scores = cosine_similarity(EMBEDDER.encode(group['business_rule'].tolist()),
                                   EMBEDDER.encode(candidates['description'].tolist()))

        for truth, row in zip(group['business_rule'], scores):
            for description, filename, score in zip(candidates['description'], candidates['filename'], row):
                rows.append({'file': doc, 'ground_truth': truth, 'business_rule': description,
                             'rule_file': filename, 'cosine': score})

    return pd.DataFrame(rows, columns = ['file', 'ground_truth', 'business_rule', 'rule_file', 'cosine'])

def top_k_index(paired: pd.DataFrame, k: int = TOP_K) -> pd.Index:
    #Index of each ground truth's k highest-cosine pairs
    return (paired.sort_values('cosine', ascending = False)
                  .groupby(['file', 'ground_truth'], sort = False)
                  .head(k).index)

def llm_results(references: list, hypotheses: list) -> list:
    #Runs the LLM evaluator only on pairs it hasn't scored yet, then returns a result for every pair
    keys = list(zip(references, hypotheses))
    missing = list(dict.fromkeys(key for key in keys if key not in LLM_CACHE))     #new pairs, duplicates removed
    print(f'  LLM: {len(missing)} new pairs to score, {len(keys) - len(missing)} already cached')

    if missing:
        results = sim_app.batch([{'statement_a': a, 'statement_b': b} for a, b in missing],
                                config = {'max_concurrency': LLM_CONCURRENCY})
        LLM_CACHE.update(zip(missing, results))

    return [LLM_CACHE[key] for key in keys]

def score_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    #Adds METEOR and ROUGE to every pair, and BERT and LLM scores to each ground truth's top TOP_K pairs
    paired = pairs.dropna(subset = ['business_rule'])
    if paired.empty:
        return pairs

    #Cheap metrics: every pair
    references, hypotheses = paired['ground_truth'].tolist(), paired['business_rule'].tolist()
    scores = pd.DataFrame(index = paired.index)

    scores['meteor'] = [compute_meteor(r, h) for r, h in zip(references, hypotheses)]

    rouge = [compute_rouge(r, h) for r, h in zip(references, hypotheses)]
    for name in ('rouge1', 'rouge2', 'rougeL'):
        for part in ('precision', 'recall', 'fmeasure'):
            scores[f'{name}_{part}'] = [result[name][part] for result in rouge]

    #Expensive metrics: only the top TOP_K pairs per ground truth; the rest are left blank
    best = paired.loc[top_k_index(paired)]
    print(f'  {len(paired)} pairs total, {len(best)} in the top {TOP_K} get BERT and LLM scores')
    references, hypotheses = best['ground_truth'].tolist(), best['business_rule'].tolist()
    deep = pd.DataFrame(index = best.index)

    P, R, F1 = BERT.score(hypotheses, references, verbose = False)
    deep['bert_precision'], deep['bert_recall'], deep['bert_f1'] = P.tolist(), R.tolist(), F1.tolist()

    #LLM evaluator: one score per iteration, repeating the last one if the loop stopped early
    results = llm_results(references, hypotheses)
    passes = [[entry['score'] for entry in result['history']] for result in results]
    passes = [p + [p[-1]] * (MAX_SIM_ITERATION - len(p)) for p in passes]
    for number in range(MAX_SIM_ITERATION):
        deep[f'llm_score_{number + 1}'] = [p[number] for p in passes]
    deep['llm_score'] = [result['score'] for result in results]
    deep['llm_reason'] = [result['reason'] for result in results]

    return pairs.join(scores.join(deep))

def compare(rules_path: str, output_path: str, truth_path: str = GROUND_TRUTH) -> pd.DataFrame:
    #Compares one file of extracted business rules to the ground truths and saves the result
    truths, rules = pd.read_csv(truth_path), pd.read_csv(rules_path)
    result = score_pairs(all_pairs(truths, rules))
    result.to_csv(output_path, index = False)
    return result

if __name__ == '__main__':
    for prompt_format in PROMPT_FORMATS:
        for iteration in range(MAX_BR_ITERATION + 1):
            rules_path = f'{OUTPUT_FOLDER}/trial2/{prompt_format}/br{iteration}.csv'
            if not Path(rules_path).exists():
                print(f'{rules_path} not found, skipped')
                continue

            print(f'{prompt_format} iteration {iteration}:')
            result = compare(rules_path, f'{OUTPUT_FOLDER}/trial2/{prompt_format}/comparison{iteration}.csv')
            unpaired = result['business_rule'].isna().sum()
            print(f'  {len(result) - unpaired} pairs scored, '
                  f'{unpaired} ground truths with no rules to compare against')