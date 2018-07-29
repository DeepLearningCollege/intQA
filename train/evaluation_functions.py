"""Functions used for evaluation of model accuracy.
"""

import re
import string

from collections import Counter

import numpy as np


def get_sampled_start_and_end(start_probs, end_probs, options):
    best_prob = 0
    sampled_start = 0
    sampled_end = 0

    # TODO if start_probs is tensor, do np.random.choic in TF
    # https://stackoverflow.com/questions/41123879/numpy-random-choice-in-tensorflow
    start_pos = np.random.choice(start_probs.shape[0], 1, p=start_probs)
    max_end_pos = min(end_probs.shape[0], start_pos + options.max_search_span_range)

    truncated_end_probs = end_probs[start_pos:max_end_pos]
    norm_end_probs = truncated_end_probs / sum(truncated_end_probs)
    end_pos = start_pos + np.random.choice(len(norm_end_probs), 1, p=norm_end_probs)
    return start_pos, end_pos


def get_best_start_and_end(start_probs, end_probs, options):
    best_prob = 0
    best_start = 0
    best_end = 0
    for z in range(start_probs.shape[0]):
        start_prob = start_probs[z]
        for zz in range(z, min(end_probs.shape[0], z + options.max_search_span_range)):
            end_prob = end_probs[zz]
            prob = start_prob * end_prob
            if prob > best_prob:
                best_prob = prob
                best_start = z
                best_end = zz
    return best_start, best_end

def avg_over_list(metric_fn, predictions, ground_truths):
    avg_value = 0.0
    for i in range(len(predictions)):
        avg_value += max_over_gnd_truths(metric_fn, predictions[i], ground_truths[i])
    avg_value /= len(predictions)
    return avg_value

def max_over_gnd_truths(metric_fn, prediction, ground_truths):
    max_value = 0
    for gnd_truth in ground_truths:
        max_value = max(max_value, metric_fn(prediction, gnd_truth))
    return max_value

def f1_score(prediction, ground_truth):
    if prediction == ground_truth:
        return 1
    prediction_tokens = _normalize_answer(prediction).split()
    ground_truth_tokens = _normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return (_normalize_answer(prediction) == _normalize_answer(ground_truth))

def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
