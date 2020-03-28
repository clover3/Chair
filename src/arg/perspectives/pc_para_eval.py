import pickle
from typing import List, Tuple, NewType, Dict, Callable, Any

import numpy as np
from scipy.special import softmax

from arg.perspectives.cpid_def import CPID
from data_generator.common import get_tokenizer
from data_generator.subword_translate import Subword
from evals.tfrecord import load_tfrecord
from list_lib import lmap, left, unique_from_sorted, right
from misc_lib import group_by, average
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer

Segment = NewType('Segment', List[Subword])


def split_3segments(input_tokens: Segment) -> Tuple[Segment, Segment, Segment]:
    try:
        idx1 = input_tokens.index("[SEP]")
        idx2 = input_tokens.index("[SEP]", idx1+1)
        idx3 = input_tokens.index("[SEP]", idx2+1)
    except:
        print("Parse fail")
        raise Exception

    return input_tokens[1:idx1], input_tokens[idx1+1:idx2], input_tokens[idx2+1:idx3]


def input_tokens_to_key(input_tokens):
    claim, pers, _ = split_3segments(input_tokens)
    claim_text = " ".join(claim)
    p_text = " ".join(pers)
    key = claim_text + "_" + p_text
    return key


def load_prediction(pred_path) -> List[Tuple[str, List[float]]]:
    data = EstimatorPredictionViewer(pred_path)

    def parse_entry(entry) -> Tuple[str, float]:
        input_tokens: Segment = entry.get_tokens('input_ids')
        logits = entry.get_vector("logits")
        probs = softmax(logits)
        key = input_tokens_to_key(input_tokens)
        score = probs[1]

        return key, score

    parsed_data: List[Tuple[str, float]] = lmap(parse_entry, data)

    keys: List[str] = unique_from_sorted(left(parsed_data))
    grouped: Dict[str, List[Tuple[str, float]]] = group_by(parsed_data, lambda x: x[0])

    def fetch_scores(key):
        l = []
        for k2, score in grouped[key]:
            assert key == k2
            l.append(score)
        return key, l

    results: List[Tuple[str, List[float]]] = lmap(fetch_scores, keys)
    return results


def load_label_from_tfrecord(tfrecord_path):
    itr = load_tfrecord(tfrecord_path)
    tokenizer = get_tokenizer()

    label_d = {}
    for e in itr:
        input_ids, label_ids = e
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        key = input_tokens_to_key(input_tokens)
        if key in label_d:
            assert label_d[key] == label_ids
        label_d[key] = label_ids

    return label_d


def load_label(label_path):
    return pickle.load(open(label_path, "rb"))


def pc_eval(pred_path, label_path, option="avg"):
    keys, reduced_scores = get_scores(option, pred_path)

    predictions = zip(keys, reduced_scores)
    labels_d: Dict[str, int] = load_label(label_path)

    label_list: List[int] = lmap(lambda x: labels_d[x], keys)

    pred = lmap(lambda x: int(x > 0.5), reduced_scores)

    num_correct = np.count_nonzero(np.equal(pred, label_list))
    print("Acc : ", num_correct / len(label_list))


def get_scores(option, pred_path) -> Tuple[List[str], List[float]]:
    raw_predictions: List[Tuple[str, List[float]]] = load_prediction(pred_path)
    if option == "avg":
        reducer: Callable[[List[Any]], float] = average
    elif option == "max":
        reducer: Callable[[List[Any]], float] = max
    else:
        assert False
    keys = left(raw_predictions)
    reduced_scores = lmap(reducer, right(raw_predictions))
    return keys, reduced_scores


def get_cpid_score(pred_path: str,
                   cpid_resolute_d: Dict[str, CPID],
                   option="avg") -> Dict[CPID, float]:
    keys, reduced_scores = get_scores(option, pred_path)
    cpid_list = lmap(lambda x:cpid_resolute_d[x], keys)
    return dict(zip(cpid_list, reduced_scores))


