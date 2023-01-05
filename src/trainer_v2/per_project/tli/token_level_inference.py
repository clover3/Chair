from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import scipy.special

from trainer_v2.chair_logging import c_log

Numpy2D = np.array
Numpy1D = np.array


def token_level_vector_attribution(
        scores: List[List[float]],
        intervals: List[Tuple[int, int]],
        n_seq=None
) -> np.array:
    scores_building: Dict[int, List[List[float]]] = defaultdict(list)

    score_len = 3
    for s, (st, ed) in zip(scores, intervals):
        score_len = len(s)
        for i in range(st, ed):
            scores_building[i].append(s)
    if n_seq is None:
        n_seq = max(scores_building.keys()) + 1

    vector_arrays: List[np.array] = []
    for i in range(n_seq):
        probs_list = scores_building[i]
        if not probs_list:
            vector_arrays.append(np.zeros([score_len]))
        else:
            probs_np = np.array(probs_list)  # [N, 3]
            vector_arrays.append(np.mean(probs_np, axis=0))
    return np.stack(vector_arrays, axis=0)


class TokenLevelInference:
    def __init__(self, nli_predict_fn, enum_subseq):
        self.nli_predict_fn = nli_predict_fn
        self.enum_subseq = enum_subseq

    def do_both_way(self, sent1, sent2) -> Tuple[np.array, np.array]:
        tli1 = self.do_one(sent1, sent2)
        tli2 = self.do_one(sent2, sent1)
        return tli1, tli2

    def do_one(self, prem, hypo) -> np.array:
        h_tokens = hypo.split()
        payload: List[Tuple[str, str]] = []
        payload_info: List[Tuple[int, int]] = []
        c_log.debug("do_one_side")
        subseq_list: List[Tuple[int, int]] = list(self.enum_subseq(len(h_tokens)))
        for st, ed in subseq_list:
            h = " ".join(h_tokens[st:ed])
            payload.append((prem, h))
            payload_info.append((st, ed))

        c_log.debug("{} payloads".format(len(payload)))
        preds: List[List[float]] = self.nli_predict_fn(payload)
        c_log.debug("Recieved response")
        pred_d = {}
        for pred, info in zip(preds, payload_info):
            st, ed = info
            pred_d[(st, ed)] = pred

        return token_level_vector_attribution(preds, payload_info)

    def do_batch(self, pairs: List[Tuple[str, str]]) -> List[Numpy2D]:
        payload: List[Tuple[str, str]] = []
        for prem, hypo in pairs:
            h_tokens = hypo.split()
            subseq_list: List[Tuple[int, int]] = list(self.enum_subseq(len(h_tokens)))
            for st, ed in subseq_list:
                h = " ".join(h_tokens[st:ed])
                payload.append((prem, h))

        payload = list(set(payload))
        c_log.debug("TokenLevelInference::do_batch() - {} payloads".format(len(payload)))
        preds: List[List[float]] = self.nli_predict_fn(payload)
        c_log.debug("Recieved response")

        preds_d: Dict[Tuple[str, str], List[float]] = {}
        for pred, pair in zip(preds, payload):
            preds_d[pair] = pred

        out_attrib_list = []
        for prem, hypo in pairs:
            h_tokens = hypo.split()
            subseq_list: List[Tuple[int, int]] = list(self.enum_subseq(len(h_tokens)))
            preds_for_this_pair = []
            payload_info_for_this_pair = []
            for st, ed in subseq_list:
                h = " ".join(h_tokens[st:ed])
                preds_for_this_pair.append(preds_d[prem, h])
                payload_info_for_this_pair.append((st, ed))

            atrib: np.array = token_level_vector_attribution(preds_for_this_pair, payload_info_for_this_pair)
            out_attrib_list.append(atrib)
        return out_attrib_list

    def do_batch_return_dict(self, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Numpy2D]:
        outputs = self.do_batch(pairs)
        tli_dict: Dict[Tuple[str, str], Numpy2D] = dict(zip(pairs, outputs))
        return tli_dict


class TokenLevelInferenceExclusion:
    def __init__(self, nli_predict_fn, enum_subseq_ex):
        self.nli_predict_fn = nli_predict_fn
        self.enum_subseq_ex = enum_subseq_ex

    def do_batch(
            self, pairs: List[Tuple[str, str, List[int]]]) -> List[Numpy2D]:
        payload: List[Tuple[str, str]] = []
        for prem, hypo, ex_mask in pairs:
            h_tokens = hypo.split()
            subseq_list: List[Tuple[int, int]] = list(self.enum_subseq_ex(len(h_tokens), ex_mask))
            for st, ed in subseq_list:
                h = " ".join(h_tokens[st:ed])
                payload.append((prem, h))

        payload = list(set(payload))
        c_log.debug("TokenLevelInference::do_batch() - {} payloads".format(len(payload)))
        preds: List[List[float]] = self.nli_predict_fn(payload)
        c_log.debug("Recieved response")

        preds_d: Dict[Tuple[str, str], List[float]] = {}
        for pred, pair in zip(preds, payload):
            preds_d[pair] = pred

        out_attrib_list = []
        for prem, hypo, ex_mask in pairs:
            h_tokens = hypo.split()
            subseq_list: List[Tuple[int, int]] = list(self.enum_subseq_ex(len(h_tokens), ex_mask))
            preds_for_this_pair = []
            payload_info_for_this_pair = []
            for st, ed in subseq_list:
                h = " ".join(h_tokens[st:ed])
                preds_for_this_pair.append(preds_d[prem, h])
                payload_info_for_this_pair.append((st, ed))

            atrib: np.array = token_level_vector_attribution(
                preds_for_this_pair,
                payload_info_for_this_pair,
                len(h_tokens)
            )
            out_attrib_list.append(atrib)
        return out_attrib_list

    def do_batch_return_dict(self, pairs: List[Tuple[str, str, List[int]]]) -> Dict[Tuple[str, str, str], Numpy2D]:
        outputs = self.do_batch(pairs)
        pairs_key = [(s1, s2, mask_to_str(mask)) for s1, s2, mask in pairs]
        tli_dict: Dict[Tuple[str, str, str], Numpy2D] = dict(zip(pairs_key, outputs))
        return tli_dict


def mask_to_str(i_arr):
    return "".join(map(str, i_arr))


def max_reduce_then_softmax(tli_p_h: np.array) -> np.array:
    raw_logits = np.max(tli_p_h, axis=0)  # [3]
    probs = scipy.special.softmax(raw_logits)
    return probs


# Input: [N, 3]
# Output: [3]
def nc_max_e_avg_reduce_then_softmax(tli_p_h: np.array) -> np.array:
    e_logit = np.mean(tli_p_h[:, 0], axis=0)
    n_logit = np.max(tli_p_h[:, 1], axis=0)
    c_logit = np.max(tli_p_h[:, 2], axis=0)
    raw_logits = np.stack([e_logit, n_logit, c_logit])
    probs = scipy.special.softmax(raw_logits)
    return probs