from collections import Counter
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np


@dataclass
class EffectiveRankedListInfo:
    qid: str
    # doc_ids and scores are paired and sorted by scores
    doc_ids: List[str]
    scores: List[float]
    base_rank: int  # number of relevant document before this ranked list
    rel_idx: int  # index of relevant document in the list


def ed(arr):
    return np.expand_dims(arr, 1)


#NI = NotImplemented

def fetch(terms: List[str], tf: Counter):
    return [tf[t] for t in terms]


class ScoringModel:
    def __init__(self, k1, b, avdl, qtw):
        self.k1 = k1
        self.b = b
        self.avdl = avdl
        self.qtw = qtw

    def get_score(self, tf_arr: np.array, dl: int):
        k1 = self.k1
        b = self.b
        avdl = self.avdl
        denom = tf_arr + k1 * tf_arr
        nom = tf_arr + k1 * ((1-b) + b * dl / avdl)
        return np.divide(denom, nom) * self.qtw


def is_pos(arr):
    return np.less(0, arr).astype(int)


def is_neg(arr):
    return np.less(arr, 0).astype(int)


def compute_alignment_gains(
        e_ranked_list_list: List[EffectiveRankedListInfo],
        doc_id_to_tf: Dict[str, Counter],
        qid_doc_id_to_target_tf: Dict[Tuple[str, str], int],
        term_targets: List[str],
        get_score: Callable[[np.array, int], np.array],
        weight=np.array([0.1])) -> np.array:
    """
    :param e_ranked_list_list:
    :param doc_id_to_tf:
    :param qid_doc_id_to_target_tf: term frequency of 'when'
    :param term_targets: the terms that we want to expand
    :param get_score: compute bm25 like score per query term
    :param weight:
    :return: gain arrays of shape [len(qid), len(term_target)]
    """

    gain_per_qid = []
    for e_ranked_list in e_ranked_list_list:
        qid = e_ranked_list.qid
        # We have only one term per query as target
        # shape [len(effective_docs)]
        e_docs = e_ranked_list.doc_ids

        # For documents whose scores are close to relevant documents
        delta_scores_array = []
        for doc_id in e_docs:
            doc_tf = doc_id_to_tf[doc_id]
            dl = sum(doc_tf.values())

            # independent of term_targets
            TF_orig: int = qid_doc_id_to_target_tf[qid, doc_id]

            # This is the most costly op which cannot be optimized
            target_term_tf: List[int] = fetch(term_targets, doc_tf)
            TF_new = target_term_tf * weight + TF_orig

            orig_score = get_score(np.array([TF_orig]), dl)
            new_score = get_score(TF_new, dl)
            delta_scores = new_score - orig_score
            delta_scores_array.append(delta_scores)

        # Shape = [len(effective_docs) , len(term_targets)]
        #  e.g.,  [80, 300]
        delta_scores_arr_np = np.stack(delta_scores_array, axis=0)
        new_scores = ed(e_ranked_list.scores) + delta_scores_arr_np  # ed = expand_dims
        rel_doc_new_score = new_scores[e_ranked_list.rel_idx]
        pairwise_pref: np.array = new_scores - rel_doc_new_score

        # Shape [len(term_targets)]
        n_doc_rank_higher = np.sum(is_pos(pairwise_pref), axis=0)
        n_doc_rank_lower = np.sum(is_neg(pairwise_pref), axis=0)

        error = n_doc_rank_lower + n_doc_rank_higher + 1 - len(e_ranked_list.scores)
        if not np.all(error <= 1):
            raise Exception()

        n_doc_rank_higher_orig = e_ranked_list.rel_idx
        orig_rank = e_ranked_list.base_rank + n_doc_rank_higher_orig
        new_rank = e_ranked_list.base_rank + n_doc_rank_higher
        abs_rank_change = new_rank - orig_rank

        def rr(r):
            return 1 / (r+1)
        rr_change = rr(new_rank) - rr(orig_rank)
        # Currently gain is absolute rank change, but we could use reciprocal rank change.
        gain_per_qid.append(abs_rank_change)
    return np.stack(gain_per_qid, axis=0)



