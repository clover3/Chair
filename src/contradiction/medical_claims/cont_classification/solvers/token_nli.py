from typing import List, Callable, Dict, Tuple
import numpy as np
from contradiction.medical_claims.cont_classification.defs import ContClassificationProbabilityScorer, ContProblem
from trainer_v2.per_project.tli.tli_visualize import til_to_table
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_1, enum_subseq_136, enum_subseq_136_ex
from trainer_v2.per_project.tli.token_level_inference import Numpy2D, Numpy1D, TokenLevelInference, \
    TokenLevelInferenceExclusion, mask_to_str, nc_max_e_avg_reduce_then_softmax
from explain.pairing.run_visualizer.show_cls_probe import NLIVisualize
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig, get_pep_cache_client
import scipy.special

from visualize.html_visual import HtmlVisualizer


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
        self.html = HtmlVisualizer("tli_debug.html")

    def get_target_label(self, probs: np.array) -> float:
        return scipy.special.softmax(probs[1:3])[1]
        # pred = np.argmax(probs)
        # if pred == self.target_label:
        #     return 1
        # else:
        #     return 0
        #
        # return probs[self.target_label]

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
            self.html.write_paragraph("Q: {}".format(p.question))
            self.log_tli("q c1", p.claim1_text, tli_q_c1)
            self.log_tli("q c2", p.claim2_text, tli_q_c2)
            self.log_tli("c1 c2", p.claim2_text, tli_c1_c2)
            self.log_tli("c2 c1", p.claim1_text, tli_c2_c1)

            # Step 2. Identify condition tokens in claims
            #     - If a token in the claim is not entailed by the question, then it is condition
            condition_c1: Numpy1D = tli_q_c1[:, 1]  # [len(C1)]
            condition_c2: Numpy1D = tli_q_c2[:, 1]  # [len(C2)]

            def apply_weight(v: Numpy2D, weights: Numpy1D):
                return np.multiply(v, np.expand_dims(weights, 1))

            # Step 3. Check if two claims are contradictory if condition tokens are excluded
            tli_c1_c2_weighted: Numpy2D = apply_weight(tli_c1_c2, 1 - condition_c2)
            tli_c2_c1_weighted: Numpy2D = apply_weight(tli_c2_c1, 1 - condition_c1)
            self.log_tli("c1 c2 weighted", p.claim2_text, tli_c1_c2_weighted)
            self.log_tli("c2 c1 weighted", p.claim1_text, tli_c2_c1_weighted)

            probs1: Numpy1D = self.combine_tli(tli_c1_c2_weighted)
            probs2: Numpy1D = self.combine_tli(tli_c2_c1_weighted)
            probs: Numpy1D = (probs1 + probs2) / 2
            label_probs: float = self.get_target_label(probs)
            self.html.write_paragraph("Final probs: {}".format(
                NLIVisualize.make_prediction_summary_str(probs)))
            out_scores.append(label_probs)

        return out_scores

    def log_tli(self, msg, hypo, tli):
        table = til_to_table(hypo, tli)
        self.html.write_paragraph(msg)
        self.html.write_table(table)


class TokenLevelSolverWOCondition(ContClassificationProbabilityScorer):
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
        return scipy.special.softmax(probs[1:3])[1]

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        tli_payload = []
        for p in problems:
            tli_payload.append((p.claim1_text, p.claim2_text))
            tli_payload.append((p.claim2_text, p.claim1_text))

        c_log.debug("Computing TLI...")
        tli_d: Dict[Tuple[str, str], Numpy2D] = self.tli_module.do_batch_return_dict(tli_payload)
        c_log.debug("Computing TLI Done")

        out_scores: List[float] = []
        for p in problems:
            tli_c1_c2 = tli_d[p.claim1_text, p.claim2_text]
            tli_c2_c1 = tli_d[p.claim2_text, p.claim1_text]

            probs1: Numpy1D = self.combine_tli(tli_c1_c2)
            probs2: Numpy1D = self.combine_tli(tli_c2_c1)
            probs: Numpy1D = (probs1 + probs2) / 2
            label_probs: float = self.get_target_label(probs)
            out_scores.append(label_probs)

        return out_scores


class TokenLevelSolverExEnum(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 combine_tli: Callable[[Numpy2D], Numpy1D],
                 enum_subseq: Callable,
                 enum_subseq_ex: Callable,
                 target_label,
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.tli_module = TokenLevelInference(nli_predict_fn, enum_subseq)
        self.tli_ex_module = TokenLevelInferenceExclusion(nli_predict_fn, enum_subseq_ex)
        self.target_label = target_label
        self.combine_tli: Callable[[Numpy2D], Numpy1D] = combine_tli
        self.html = HtmlVisualizer("tli_ex_debug.html")

    def get_target_label(self, probs: np.array) -> float:
        pred = np.argmax(probs)
        if pred == self.target_label:
            return 1
        else:
            return 0
        # return probs[self.target_label]

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:

        tli_payload = []
        for p in problems:
            tli_payload.append((p.question + "? Yes", p.claim1_text))
            tli_payload.append((p.question + "? Yes", p.claim2_text))
            tli_payload.append((p.question + "? No", p.claim1_text))
            tli_payload.append((p.question + "? No", p.claim2_text))

        c_log.debug("Computing TLI...")
        tli_d: Dict[Tuple[str, str], Numpy2D] = self.tli_module.do_batch_return_dict(tli_payload)
        c_log.debug("Computing TLI Done")

        out_scores: List[float] = []

        def int_mask(float_arr):
            return [1 if f >= 0.5 else 0 for f in float_arr]

        def get_condition_mask(p):
            tli_q_c1_ys = tli_d[p.question + "? Yes", p.claim1_text]
            tli_q_c2_ys = tli_d[p.question + "? Yes", p.claim2_text]
            tli_q_c1_no = tli_d[p.question + "? No", p.claim1_text]
            tli_q_c2_no = tli_d[p.question + "? No", p.claim2_text]

            # Step 2. Identify condition tokens in claims
            #     - If a token in the claim is not entailed by the question, then it is condition
            ex_mask1 = int_mask(np.minimum(tli_q_c1_ys[:, 1], tli_q_c1_no[:, 1]))  # [len(C1)]
            ex_mask2 = int_mask(np.minimum(tli_q_c2_ys[:, 1], tli_q_c2_no[:, 1]))  # [len(C1)]
            return ex_mask1, ex_mask2

        tli_ex_payload = []
        for p in problems:
            ex_mask1, ex_mask2 = get_condition_mask(p)
            tli_ex_payload.append((p.claim1_text, p.claim2_text, ex_mask2))
            tli_ex_payload.append((p.claim2_text, p.claim1_text, ex_mask1))

        tli_ex_d: Dict[Tuple[str, str, str], Numpy2D] = self.tli_ex_module.do_batch_return_dict(tli_ex_payload)

        # Step 3. Check if two claims are contradictory if condition tokens are excluded
        for p, tli_ex in zip(problems, tli_ex_payload):
            ex_mask1, ex_mask2 = get_condition_mask(p)

            tli_q_c1_ys = tli_d[p.question + "? Yes", p.claim1_text]
            tli_q_c2_ys = tli_d[p.question + "? Yes", p.claim2_text]
            tli_q_c1_no = tli_d[p.question + "? No", p.claim1_text]
            tli_q_c2_no = tli_d[p.question + "? No", p.claim2_text]
            ex_mask1_str = mask_to_str(ex_mask1)  # [len(C1)]
            ex_mask2_str = mask_to_str(ex_mask2)  # [len(C1)]

            tli_c1_c2 = tli_ex_d[p.claim1_text, p.claim2_text, ex_mask2_str]
            tli_c2_c1 = tli_ex_d[p.claim2_text, p.claim1_text, ex_mask1_str]
            self.html.write_paragraph("Q: {}".format(p.question))
            self.log_tli("q c1 ys", p.claim1_text, tli_q_c1_ys)
            self.log_tli("q c2 ys", p.claim2_text, tli_q_c2_ys)
            self.log_tli("q c1 no", p.claim1_text, tli_q_c1_no)
            self.log_tli("q c2 no", p.claim2_text, tli_q_c2_no)
            self.log_tli("c1 c2", p.claim2_text, tli_c1_c2)
            self.log_tli("c2 c1", p.claim1_text, tli_c2_c1)

            probs1: Numpy1D = self.combine_tli(tli_c1_c2)
            probs2: Numpy1D = self.combine_tli(tli_c2_c1)
            probs: Numpy1D = (probs1 + probs2) / 2
            label_probs: float = self.get_target_label(probs)
            self.html.write_paragraph("Final probs: {}".format(
                NLIVisualize.make_prediction_summary_str(probs)))

            out_scores.append(label_probs)

        return out_scores

    def log_tli(self, msg, hypo, tli):
        table = til_to_table(hypo, tli)
        self.html.write_paragraph(msg)
        self.html.write_table(table)


class TokenLevelSolverExEnum2(ContClassificationProbabilityScorer):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 combine_tli: Callable[[Numpy2D], Numpy1D],
                 enum_subseq: Callable,
                 enum_subseq_ex: Callable,
                 target_label,
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.tli_module = TokenLevelInference(nli_predict_fn, enum_subseq)
        self.tli_ex_module = TokenLevelInferenceExclusion(nli_predict_fn, enum_subseq_ex)
        self.target_label = target_label
        self.combine_tli: Callable[[Numpy2D], Numpy1D] = combine_tli
        self.html = HtmlVisualizer("tli_ex_debug.html")

    def get_target_label(self, probs: np.array) -> float:
        pred = np.argmax(probs)
        if pred == self.target_label:
            return 1
        else:
            return 0
        # return probs[self.target_label]

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:

        tli_payload = []
        for p in problems:
            tli_payload.append((p.question + "? Yes", p.claim1_text))
            tli_payload.append((p.question + "? Yes", p.claim2_text))
            tli_payload.append((p.question + "? No", p.claim1_text))
            tli_payload.append((p.question + "? No", p.claim2_text))

        c_log.debug("Computing TLI...")
        tli_d: Dict[Tuple[str, str], Numpy2D] = self.tli_module.do_batch_return_dict(tli_payload)
        c_log.debug("Computing TLI Done")

        out_scores: List[float] = []

        def int_mask(float_arr):
            return [1 if f >= 0.5 else 0 for f in float_arr]

        def get_condition_mask(p):
            tli_q_c1_ys = tli_d[p.claim1_text, p.question + "? Yes"]
            tli_q_c2_ys = tli_d[p.claim2_text, p.question + "? Yes"]
            tli_q_c1_no = tli_d[p.claim1_text, p.question + "? No"]
            tli_q_c2_no = tli_d[p.claim2_text, p.question + "? No"]

            # Step 2. Identify condition tokens in claims
            #     - If a token in the claim is not entailed by the question, then it is condition
            ex_mask1 = int_mask(np.minimum(tli_q_c1_ys[:, 1], tli_q_c1_no[:, 1]))  # [len(C1)]
            ex_mask2 = int_mask(np.minimum(tli_q_c2_ys[:, 1], tli_q_c2_no[:, 1]))  # [len(C1)]
            return ex_mask1, ex_mask2

        tli_ex_payload = []
        for p in problems:
            ex_mask1, ex_mask2 = get_condition_mask(p)
            tli_ex_payload.append((p.claim1_text, p.claim2_text, ex_mask2))
            tli_ex_payload.append((p.claim2_text, p.claim1_text, ex_mask1))

        tli_ex_d: Dict[Tuple[str, str, str], Numpy2D] = self.tli_ex_module.do_batch_return_dict(tli_ex_payload)

        # Step 3. Check if two claims are contradictory if condition tokens are excluded
        for p, tli_ex in zip(problems, tli_ex_payload):
            ex_mask1, ex_mask2 = get_condition_mask(p)

            tli_q_c1_ys = tli_d[p.claim1_text, p.question + "? Yes"]
            tli_q_c2_ys = tli_d[p.claim2_text, p.question + "? Yes", ]
            tli_q_c1_no = tli_d[p.claim1_text, p.question + "? No"]
            tli_q_c2_no = tli_d[p.claim2_text, p.question + "? No"]
            ex_mask1_str = mask_to_str(ex_mask1)  # [len(C1)]
            ex_mask2_str = mask_to_str(ex_mask2)  # [len(C1)]

            tli_c1_c2 = tli_ex_d[p.claim1_text, p.claim2_text, ex_mask2_str]
            tli_c2_c1 = tli_ex_d[p.claim2_text, p.claim1_text, ex_mask1_str]
            self.html.write_paragraph("Q: {}".format(p.question))
            self.log_tli("q c1 ys", p.claim1_text, tli_q_c1_ys)
            self.log_tli("q c2 ys", p.claim2_text, tli_q_c2_ys)
            self.log_tli("q c1 no", p.claim1_text, tli_q_c1_no)
            self.log_tli("q c2 no", p.claim2_text, tli_q_c2_no)
            self.log_tli("c1 c2", p.claim2_text, tli_c1_c2)
            self.log_tli("c2 c1", p.claim1_text, tli_c2_c1)

            probs1: Numpy1D = self.combine_tli(tli_c1_c2)
            probs2: Numpy1D = self.combine_tli(tli_c2_c1)
            probs: Numpy1D = (probs1 + probs2) / 2
            label_probs: float = self.get_target_label(probs)
            self.html.write_paragraph("Final probs: {}".format(
                NLIVisualize.make_prediction_summary_str(probs)))

            out_scores.append(label_probs)

        return out_scores

    def log_tli(self, msg, hypo, tli):
        table = til_to_table(hypo, tli)
        self.html.write_paragraph(msg)
        self.html.write_table(table)


class TokenLevelSolverExDirect(ContClassificationProbabilityScorer):
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
        self.html = HtmlVisualizer("tli_ex_direct_debug.html")

    def get_target_label(self, probs: np.array) -> float:
        return probs[self.target_label]

    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        tli_payload = []
        for p in problems:
            tli_payload.append((p.question + "? Yes", p.claim1_text))
            tli_payload.append((p.question + "? Yes", p.claim2_text))
            tli_payload.append((p.question + "? No", p.claim1_text))
            tli_payload.append((p.question + "? No", p.claim2_text))

        c_log.debug("Computing TLI...")
        tli_d: Dict[Tuple[str, str], Numpy2D] = self.tli_module.do_batch_return_dict(tli_payload)
        c_log.debug("Computing TLI Done")

        out_scores: List[float] = []

        def int_mask(float_arr):
            return [1 if f >= 0.5 else 0 for f in float_arr]

        def get_condition_mask(p):
            tli_q_c1_ys = tli_d[p.question + "? Yes", p.claim1_text]
            tli_q_c2_ys = tli_d[p.question + "? Yes", p.claim2_text]
            tli_q_c1_no = tli_d[p.question + "? No", p.claim1_text]
            tli_q_c2_no = tli_d[p.question + "? No", p.claim2_text]

            # Step 2. Identify condition tokens in claims
            #     - If a token in the claim is not entailed by the question, then it is condition
            ex_mask1 = int_mask(np.minimum(tli_q_c1_ys[:, 1], tli_q_c1_no[:, 1]))  # [len(C1)]
            ex_mask2 = int_mask(np.minimum(tli_q_c2_ys[:, 1], tli_q_c2_no[:, 1]))  # [len(C1)]
            return ex_mask1, ex_mask2

        def remove_condition(text, mask):
            tokens = text.split()
            new_tokens = []
            is_mask_before = False
            for exclude, t in zip(mask, tokens):
                if exclude:
                    if not is_mask_before:
                        new_tokens.append("[MASK]")
                        is_mask_before = True
                else:
                    is_mask_before = False
                    new_tokens.append(t)
            return " ".join(new_tokens)

        pair_payload = []
        for p in problems:
            ex_mask1, ex_mask2 = get_condition_mask(p)
            claim1_text_no_cond = remove_condition(p.claim1_text, ex_mask1)
            claim2_text_no_cond = remove_condition(p.claim2_text, ex_mask2)
            pair_payload.append((claim1_text_no_cond, claim2_text_no_cond))
            self.html.write_paragraph("Old P: " + p.claim1_text)
            self.html.write_paragraph("Old H: " + p.claim2_text)
            self.html.write_paragraph("New P: " + claim1_text_no_cond)
            self.html.write_paragraph("New H: " + claim2_text_no_cond)

        preds = self.nli_predict_fn(pair_payload)
        preds_d = dict(zip(pair_payload, preds))

        # Step 3. Check if two claims are contradictory if condition tokens are excluded
        for p in problems:
            ex_mask1, ex_mask2 = get_condition_mask(p)
            claim1_text_no_cond = remove_condition(p.claim1_text, ex_mask1)
            claim2_text_no_cond = remove_condition(p.claim2_text, ex_mask2)
            key = claim1_text_no_cond, claim2_text_no_cond
            probs = preds_d[key]
            label_probs: float = self.get_target_label(probs)
            self.html.write_paragraph("Final probs: {}".format(
                NLIVisualize.make_prediction_summary_str(probs)))

            out_scores.append(label_probs)

        return out_scores


def get_token_level_inf_classifier(run_name):
    nli_predict_fn = get_pep_cache_client()
    if run_name == "tnli1":
        classifier = TokenLevelSolver(
            nli_predict_fn,
            nc_max_e_avg_reduce_then_softmax,
            enum_subseq_136,
            2
        )
    elif run_name == "tnli2":
        classifier = TokenLevelSolver(
            nli_predict_fn,
            nc_max_e_avg_reduce_then_softmax,
            enum_subseq_1,
            2
        )
    elif run_name == "tnli3":
        classifier = TokenLevelSolverWOCondition(
            nli_predict_fn,
            nc_max_e_avg_reduce_then_softmax,
            enum_subseq_136,
            2
        )
    elif run_name == "tnli4":
        classifier = TokenLevelSolverExEnum(
            nli_predict_fn,
            nc_max_e_avg_reduce_then_softmax,
            enum_subseq_136,
            enum_subseq_136_ex,
            2
        )
    elif run_name == "tnli5":
        classifier = TokenLevelSolverExDirect(
            nli_predict_fn,
            nc_max_e_avg_reduce_then_softmax,
            enum_subseq_136,
            2
        )
    else:
        raise KeyError
    return classifier
