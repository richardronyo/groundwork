"""
compare.py

Compares each generated business rule to every ground truth rule that
belongs to the .rst file(s) its source code file maps to, using the
metrics from evaluators.py. Writes one row per (generated rule, ground
truth rule) pair.
"""

import os

import pandas as pd

from evaluators import (
    compute_meteor,
    compute_rouge,
    compute_bert,
    compute_cosine,
    CODE_DOCUMENTATION_MAP,
)


def load_groundtruth(path: str = 'groundtruth.csv') -> pd.DataFrame:
    """Loads groundtruth.csv. Columns: file, business_rule."""
    return pd.read_csv(path)


def load_generated(path: str) -> pd.DataFrame:
    """Loads a results_<prompt_type>.csv. Columns: file, business_rule."""
    return pd.read_csv(path)


def build_pairs(generated_df: pd.DataFrame, groundtruth_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each generated rule, looks up the .rst file(s) its source code
    file maps to, then pairs it with every ground truth rule belonging
    to those .rst file(s).
    """
    generated_df = generated_df.copy()

    generated_df['doc_file'] = generated_df['file'].apply(
        lambda code_file: CODE_DOCUMENTATION_MAP.get(os.path.basename(code_file), [])
    )

    # one row per (generated rule, mapped doc file); drops rows with no mapping
    generated_df = generated_df.explode('doc_file').dropna(subset=['doc_file'])

    pairs = generated_df.merge(
        groundtruth_df,
        left_on='doc_file',
        right_on='file',
        suffixes=('_generated', '_groundtruth'),
    )

    pairs = pairs.rename(columns={
        'business_rule_generated': 'generated_rule',
        'business_rule_groundtruth': 'ground_truth_rule',
    })

    return pairs[['generated_rule', 'ground_truth_rule']].reset_index(drop=True)


def compute_metrics(row: pd.Series) -> pd.Series:
    """Runs every evaluator on one (generated, ground truth) pair and flattens the result."""
    generated_rule = row['generated_rule']
    ground_truth_rule = row['ground_truth_rule']

    meteor = compute_meteor(ground_truth_rule, generated_rule)
    rouge = compute_rouge(ground_truth_rule, generated_rule)
    bert = compute_bert(ground_truth_rule, generated_rule)
    cosine = compute_cosine(ground_truth_rule, generated_rule)

    return pd.Series({
        'meteor': meteor,
        'rouge1_precision': rouge['rouge1']['precision'],
        'rouge1_recall': rouge['rouge1']['recall'],
        'rouge1_fmeasure': rouge['rouge1']['fmeasure'],
        'rouge2_precision': rouge['rouge2']['precision'],
        'rouge2_recall': rouge['rouge2']['recall'],
        'rouge2_fmeasure': rouge['rouge2']['fmeasure'],
        'rougeL_precision': rouge['rougeL']['precision'],
        'rougeL_recall': rouge['rougeL']['recall'],
        'rougeL_fmeasure': rouge['rougeL']['fmeasure'],
        # bert_scorer returns tensors; pull out the scalar float
        'bert_precision': bert['precision'].item(),
        'bert_recall': bert['recall'].item(),
        'bert_f1': bert['F1'].item(),
        'cosine_similarity': float(cosine),
    })


def compare_file(generated_path: str, groundtruth_df: pd.DataFrame, output_path: str) -> None:
    """
    Compares every generated rule in generated_path against every ground
    truth rule belonging to the .rst file(s) its source file maps to, and
    writes the results to output_path.
    """
    generated_df = load_generated(generated_path)
    pairs = build_pairs(generated_df, groundtruth_df)

    metrics = pairs.apply(compute_metrics, axis=1)
    result = pd.concat([pairs, metrics], axis=1)

    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    groundtruth_df = load_groundtruth('groundtruth.csv')

    part1 = ['general', 'user_story', 'definition']

    for prompt_type in part1:
        compare_file(
            generated_path=f'outputs/{prompt_type}.csv',
            groundtruth_df=groundtruth_df,
            output_path=f'outputs/comparison_{prompt_type}.csv',
        )