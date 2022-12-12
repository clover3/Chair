from collections import defaultdict
from typing import List, NamedTuple
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
from contradiction.medical_claims.cont_classification.defs import ContClassificationProbabilityScorer, ContProblem
from contradiction.medical_claims.retrieval.nli_system import enum_subseq_136
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_client
import scipy.special


Numpy2D = np.array
Numpy1D = np.array


# Conditionally contradictory pair.
"""
Strategy with Partial View

Step 1. Compare question and claims to check they are relevant
    - They are relevant if they entail all tokens except stopwords
    - Build token-level inference (like token_tagging)
        - Enum segments, give each tokens scores.
        
    - Relevant: weighted sum strategy
        - Use IDF weights to combine tokens
    
Step 2. Identify condition tokens in claims,
    - If a token in the claim is not entailed by the question, then it is condition
        - 
        
Step 3. Check if two claims are contradictory if condition tokens are excluded
    - 

How to convert token level inference score (TLI) into classification score.

When a sentence is split into two, 
    
Strategy with Full view
    
Step 1. only enum individual word
    
"""


def token_level_vector_attribution(
        scores: List[List[float]],
        intervals: List[Tuple[int, int]]) -> np.array:
    scores_building: Dict[int, List[List[float]]] = defaultdict(list)

    for s, (st, ed) in zip(scores, intervals):
        for i in range(st, ed):
            scores_building[i].append(s)

    n_seq = max(scores_building.keys()) + 1
    vector_arrays: List[np.array] = []
    for i in range(n_seq):
        probs_list = scores_building[i]
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


# Input: [N, 3]
# Output: [3]
def max_reduce_then_softmax(tli_p_h: np.array) -> np.array:
    raw_logits = np.max(tli_p_h, axis=0)  # [3]
    probs = scipy.special.softmax(raw_logits)
    return probs


def nc_max_e_avg_reduce_then_softmax(tli_p_h: np.array) -> np.array:
    e_logit = np.mean(tli_p_h[:, 0], axis=0)
    n_logit = np.max(tli_p_h[:, 1], axis=0)
    c_logit = np.max(tli_p_h[:, 2], axis=0)
    raw_logits = np.stack([e_logit, n_logit, c_logit])
    probs = scipy.special.softmax(raw_logits)
    return probs


class TokenLevelClassifier(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 combine_tli: Callable[[Numpy2D], Numpy1D],
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
        self.combine_tli: Callable[[Numpy2D], Numpy1D] = combine_tli

    def solve_batch(self, pair_list: List[Tuple[str, str]]) -> List[np.array]:
        out_scores = []
        for prem, hypo in pair_list:
            tli: np.array = self.tli_module.do_one(prem, hypo)
            assert len(tli) == len(hypo.split())
            probs = self.combine_tli(tli)
            out_scores.append(probs)
        return out_scores


class TokenLevelSolver(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 combine_tli: Callable[[Numpy2D], Numpy1D],
                 enum_subseq: Callable,
                 target_label,
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.tli_module = TokenLevelInference(nli_predict_fn, enum_subseq)
        self.target_label = target_label
        self.combine_tli: Callable[[Numpy2D], Numpy1D] = combine_tli

    def get_target_label(self, probs: np.array) -> float:
        return probs[self.target_label]

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        tli_payload = []
        for p in problems:
            tli_payload.append((p.question, p.claim1_text))
            tli_payload.append((p.question, p.claim2_text))
            tli_payload.append((p.claim1_text, p.claim2_text))
            tli_payload.append((p.claim2_text, p.claim1_text))

        c_log.debug("Computing TLI...")
        tli_d: Dict[Tuple[str, str], Numpy2D] = self.tli_module.do_batch_return_dict(tli_payload)
        c_log.debug("Computing TLI Done")

        out_scores: List[float] = []
        for p in problems:
            tli_q_c1 = tli_d[p.question, p.claim1_text]
            tli_q_c2 = tli_d[p.question, p.claim2_text]
            tli_c1_c2 = tli_d[p.claim1_text, p.claim2_text]
            tli_c2_c1 = tli_d[p.claim2_text, p.claim1_text]

            # Step 2. Identify condition tokens in claims
            #     - If a token in the claim is not entailed by the question, then it is condition
            condition_c1: Numpy1D = tli_q_c1[:, 1]  # [len(C1)]
            condition_c2: Numpy1D = tli_q_c2[:, 1]  # [len(C2)]

            def apply_weight(v: Numpy2D, weights: Numpy1D):
                return np.multiply(v, np.expand_dims(weights, 1))

            # Step 3. Check if two claims are contradictory if condition tokens are excluded
            tli_c1_c2_weighted: Numpy2D = apply_weight(tli_c1_c2, condition_c2)
            tli_c2_c1_weighted: Numpy2D = apply_weight(tli_c2_c1, condition_c1)

            probs1: Numpy1D = self.combine_tli(tli_c1_c2_weighted)
            probs2: Numpy1D = self.combine_tli(tli_c2_c1_weighted)
            probs: Numpy1D = (probs1 + probs2) / 2
            label_probs: float = self.get_target_label(probs)
            out_scores.append(label_probs)

        return out_scores


def get_token_level_inf_classifier():
    nli_predict_fn = get_pep_client()
    classifier = TokenLevelSolver(
        nli_predict_fn,
        nc_max_e_avg_reduce_then_softmax,
        enum_subseq_136,
        2
    )
    return classifier
