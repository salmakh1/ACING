'''
Taken from the Instruction Induction paper: https://arxiv.org/pdf/2205.10782.pdf
'''

import re
import string
from collections import Counter
import torch
import random
import numpy as np
from rouge_score import rouge_scorer, scoring

TASKS=[
    'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
    'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 
    'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',
    'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
    'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
    'translation_en-fr', 'word_in_context', 'auto_categorization', 'auto_debugging', 'ascii', 'cs_algorithms',
    'periodic_elements', 'word_sorting', 'word_unscrambling', 'odd_one_out', 'object_counting'
]

# TODO: add some more metrics here for the new tasks.

TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains', 'samsum': 'rouge_score'}
bbh_multi_choice = ['implicatures', 'question_selection', 'logical_fallacy_detection', 'presuppositions_as_nli', 'sports_understanding', 'navigate', 'epistemic_reasoning', 'causal_judgment', 'winowhy', 'ruin_names', 'snarks', 'disambiguation_qa', 'movie_recommendation']
for task in bbh_multi_choice:
    TASK_TO_METRIC[task] = 'multichoice'
default_metric = 'em'


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# def get_rouge_scores_1(prediction, ground_truth):
#     """
#     Calculate ROUGE-1, ROUGE-2, and ROUGE-3 scores between prediction and ground_truth.
#     """
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
#     scores = scorer.score(ground_truth, prediction)
#     #, scores['rouge2'].fmeasure, scores['rouge3'].fmeasure
#     return scores['rouge1'].fmeasure


def get_rouge_scores_1(prediction, ground_truth):
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-3 scores between prediction and ground_truth.
    """
    # Ensure the inputs are strings
    print("rouge1")
    prediction = str(prediction)
    ground_truth = str(ground_truth)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)

    return scores['rouge1'].fmeasure

def get_rouge_scores_2(prediction, ground_truth):
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-3 scores between prediction and ground_truth.
    """
    # Ensure the inputs are strings
    print("rouge2")
    prediction = str(prediction)
    ground_truth = str(ground_truth)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)

    return scores['rouge2'].fmeasure

def get_rouge_scores_3(prediction, ground_truth):
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-3 scores between prediction and ground_truth.
    """
    # Ensure the inputs are strings
    print("rougeL")
    prediction = str(prediction)
    ground_truth = str(ground_truth)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)

    return scores['rougeL'].fmeasure

def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    # print("prediction is ", prediction_normalized)
    # print("ground truth is ", ground_truth_normalized)

    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1


def get_multi_answer_em(prediction, answers):
    print("em")
    for answer in answers:
        print(prediction, answer)
        if get_em_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_choice_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=False)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=False)
    return ground_truth_normalized in prediction_normalized


def get_multi_choice(prediction, answers):
    for answer in answers:
        if get_multi_choice_score(prediction, answer) == 1:
            return 1
    return 0

def get_multi_answer_f1(prediction, answers):
    f1_scores = []
    for answer in answers:
        f1_scores.append(get_f1_score(prediction, answer))
    return max(f1_scores)


def get_multi_answer_exact_set(prediction, answers):
    # print(prediction)
    # print(answers)
    for answer in answers:
        if get_exact_set_score(prediction, answer) == 1:
            return 1
    return 0


def get_multi_answer_contains(prediction, answers):
    for answer in answers:
        if get_contains_score(prediction, answer) == 1:
            return 1
    return 0

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"