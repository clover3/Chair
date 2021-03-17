import random
from typing import List, Callable, Dict, Tuple

import math

from cache import load_from_pickle
from misc_lib import find_max_idx
from trec.qrel_parse import load_qrels_structured


def load_score_d():
    all_score_d = {}
    for data_name in [301, 351, 401, 601]:
        score_d = load_from_pickle("rob_seg_score_{}".format(data_name))
        all_score_d.update(score_d)
    return all_score_d


def get_selection_fn1(alpha) -> Callable[[str, str, List], List[int]]:
    qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
    all_judgement = load_qrels_structured(qrel_path)
    score_d: Dict[Tuple[str, str, int], float] = load_score_d()
    g = 0.5

    def select_fn(query_id: str, doc_id: str, segments: List) -> List[int]:
        judgement = all_judgement[query_id]
        label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
        if label:
            prediction_scores: List[float] = list([score_d[query_id, doc_id, idx] for idx in range(len(segments))])
        else:
            prediction_scores: List[float] = [1.0] + [0.1] * (len(segments)-1)

        selected_indices = []
        for idx in range(len(segments)):
            a = math.pow(g, idx)
            b = prediction_scores[idx]
            prob = a * alpha + b * (1 - alpha)

            if random.random() < prob:
                selected_indices.append(idx)
        return selected_indices

    return select_fn


def get_selection_fn2(alpha) -> Callable[[str, str, List], List[int]]:
    qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
    all_judgement = load_qrels_structured(qrel_path)
    score_d: Dict[Tuple[str, str, int], float] = load_score_d()
    g = 0.5

    def select_fn(query_id: str, doc_id: str, segments: List) -> List[int]:
        judgement = all_judgement[query_id]
        label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
        if label:
            prediction_scores: List[float] = list([score_d[query_id, doc_id, idx] for idx in range(len(segments))])
        else:
            prediction_scores: List[float] = [1.0] + [0.1] * (len(segments)-1)

        selected_indices = []
        for idx in range(len(segments)):
            a = math.pow(g, idx)
            b = prediction_scores[idx]
            prob = a + b * alpha

            if random.random() < prob:
                selected_indices.append(idx)
        return selected_indices

    return select_fn


def get_selection_fn_include_neg() -> Callable[[str, str, List], List[int]]:
    def load_score_d():
        all_score_d = {}
        for data_name in [301, 351, 401, 601]:
            score_d = load_from_pickle("robust_3A_4_4_score_d_{}".format(data_name))
            all_score_d.update(score_d)
        return all_score_d

    score_d: Dict[Tuple[str, str, int], float] = load_score_d()

    def select_fn(query_id: str, doc_id: str, segments: List) -> List[int]:
        prediction_scores: List[float] = list([score_d[query_id, doc_id, idx] for idx in range(len(segments))])
        max_idx = find_max_idx(prediction_scores, lambda x: x)

        selected_indices = []
        selected_indices.append(max_idx)
        return selected_indices

    return select_fn
