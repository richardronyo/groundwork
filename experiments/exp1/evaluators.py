from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_meteor(reference, hypothesis):
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)

    return meteor_score([reference_tokens], hypothesis_tokens)

def compute_rouge(reference, hypothese):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothese)

    rouges = ['rouge1', 'rouge2', 'rougeL']

    ans = {}

    for rouge in rouges:
        ans[rouge] = {
            'precision': scores[rouge].precision,
            'recall': scores[rouge].recall,
            'fmeasure': scores[rouge].fmeasure
        }

    return ans

def compute_bert(reference, hypothesis):
    P, R, F1 = bert_scorer([hypothesis], [reference], lang = 'en', verbose = False)

    ans = {
        'precision': P,
        'recall': R,
        'F1': F1
    }

    return ans

_model = None

def compute_cosine(reference, hypothesis):
    global _model

    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = _model.encode([hypothesis, reference])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
 
    return sim

if __name__ == "__main__":
    reference = 'The cat sat on the mat'
    hypothesis = 'The cat is sitting on the mat'

    print('Reference: ', reference)
    print('Hypothesis: ', hypothesis)

    print('METEOR SCORE: ', compute_meteor(reference, hypothesis))
    print('ROUGE SCORE: ', compute_rouge(reference, hypothesis))
    print('BERT SCORE: ', compute_bert(reference, hypothesis))
    print('COSINE SIMILARITY SCORE: ', compute_cosine(reference, hypothesis))
