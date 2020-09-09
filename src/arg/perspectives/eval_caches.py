from typing import List, Tuple, Dict, Set

from arg.perspectives.evaluate import evaluate_map, get_average_precision_list, get_correctness_list
from arg.perspectives.types import CPIDPair
from cache import load_from_pickle
from list_lib import lmap, left, lfilter, flatten
from misc_lib import SuccessCounter, average


def get_eval_candidates_from_pickle(split) -> List[Tuple[int, List[Dict]]]:
    return load_from_pickle("pc_candidates_{}".format(split))


def predict_from_dict(score_d: Dict[CPIDPair, float],
                      candidates: List[Tuple[int, List[Dict]]],
                      top_k,
                      ) -> List[Tuple[int, List[Dict]]]:
    suc_count = SuccessCounter()

    def rank(e: Tuple[int, List[Dict]]):
        cid, p_list = e
        scored_p_list: List[Dict] = []
        for p in p_list:
            pid = int(p['pid'])
            query_id = CPIDPair((cid, pid))

            if query_id in score_d:
                score = score_d[query_id]
                suc_count.suc()
            else:
                score = -2
                suc_count.fail()
            p['score'] = score
            scored_p_list.append(p)

        scored_p_list.sort(key=lambda x: x['score'], reverse=True)
        return cid, scored_p_list[:top_k]

    predictions = lmap(rank, candidates)
    print("{} found of {}".format(suc_count.get_suc(), suc_count.get_total()))
    return predictions


def eval_map(split, score_d: Dict[CPIDPair, float], debug=False):
    # load pre-computed perspectives
    candidates: List[Tuple[int, List[Dict]]] = get_eval_candidates_from_pickle(split)
    # only evalaute what's available
    valid_cids: Set[int] = set(left(score_d.keys()))
    sub_candidates: List[Tuple[int, List[Dict]]] = lfilter(lambda x: x[0] in valid_cids, candidates)
    print("{} claims are evaluated".format(len(sub_candidates)))
    predictions = predict_from_dict(score_d, sub_candidates, 50)
    return evaluate_map(predictions, debug)


def get_ap_list_from_score_d(score_d, split):
    candidates: List[Tuple[int, List[Dict]]] = get_eval_candidates_from_pickle(split)
    # only evalaute what's available
    valid_cids: Set[int] = set(left(score_d.keys()))
    sub_candidates: List[Tuple[int, List[Dict]]] = lfilter(lambda x: x[0] in valid_cids, candidates)
    print("{} claims are evaluated".format(len(sub_candidates)))
    predictions = predict_from_dict(score_d, sub_candidates, 50)
    cids = left(predictions)
    ap_list = get_average_precision_list(predictions, False)
    return ap_list, cids


def get_acc_from_score_d(score_d, split):
    predictions = extract_predictions(score_d, split)
    cids = left(predictions)
    corr_list_list = get_correctness_list(predictions, False)
    return average(flatten(corr_list_list))


def extract_predictions(score_d, split):
    candidates: List[Tuple[int, List[Dict]]] = get_eval_candidates_from_pickle(split)
    # only evalaute what's available
    valid_cids: Set[int] = set(left(score_d.keys()))
    sub_candidates: List[Tuple[int, List[Dict]]] = lfilter(lambda x: x[0] in valid_cids, candidates)
    print("{} claims are evaluated".format(len(sub_candidates)))

    def make_decisions(e: Tuple[int, List[Dict]]):
        cid, p_list = e
        decisions = []
        for p in p_list:
            pid = int(p['pid'])
            query_id = CPIDPair((cid, pid))

            if query_id in score_d:
                score = score_d[query_id]
            else:
                score = 0

            binary = 1 if score > 0.5 else 0
            decisions.append((cid, pid, binary))

        return cid, decisions

    predictions = lmap(make_decisions, candidates)
    return predictions


def get_joined_correctness(score_d, split):
    predictions = extract_predictions(score_d, split)
    cids = left(predictions)
    corr_list_list = get_correctness_list(predictions, False)
    return list(zip(cids, corr_list_list))
