from typing import List, Tuple, NewType, Dict

import numpy as np
from scipy.special import softmax

from arg.pf_common.para_eval import input_tokens_to_key
from data_generator.subword_translate import Subword
from list_lib import lmap, left, unique_from_sorted, right
from misc_lib import group_by
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer

Segment = NewType('Segment', List[Subword])


from arg.pf_common.base import DPID
from base_type import FilePath


def load_prediction(pred_path) -> List[Tuple[str, List[np.ndarray]]]:
    data = EstimatorPredictionViewer(pred_path)

    def parse_entry(entry) -> Tuple[str, np.ndarray]:
        input_tokens: Segment = entry.get_tokens('input_ids')
        logits = entry.get_vector("logits")
        probs = softmax(logits)
        key = input_tokens_to_key(input_tokens)
        return key, probs

    parsed_data: List[Tuple[str, np.ndarray]] = lmap(parse_entry, data)

    keys: List[str] = unique_from_sorted(left(parsed_data))
    grouped: Dict[str, List[Tuple[str, np.ndarray]]] = group_by(parsed_data, lambda x: x[0])

    def fetch_scores(key):
        l = []
        for k2, score in grouped[key]:
            assert key == k2
            l.append(score)
        return key, l

    results: List[Tuple[str, List[np.ndarray]]] = lmap(fetch_scores, keys)
    return results


def get_scores(option, pred_path: FilePath) -> Tuple[List[str], List[np.ndarray]]:
    raw_predictions: List[Tuple[str, List[np.ndarray]]] = load_prediction(pred_path)
    if option == "avg":
        def reducer(data: List[np.ndarray]) -> np.ndarray:
            np_arr: np.ndarray = np.array(data)
            return np_arr.mean(axis=0)
    else:
        assert False
    keys = left(raw_predictions)
    reduced_scores = lmap(reducer, right(raw_predictions))
    return keys, reduced_scores


def get_datapoint_score(pred_path: FilePath,
                   dpid_resolute_d: Dict[str, DPID],
                   option="avg") -> Dict[DPID, np.ndarray]:
    keys, reduced_scores = get_scores(option, pred_path)
    dpid_list = lmap(lambda x: dpid_resolute_d[x], keys)
    return dict(zip(dpid_list, reduced_scores))



