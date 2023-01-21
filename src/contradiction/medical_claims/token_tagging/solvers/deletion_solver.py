import scipy.special
from typing import Tuple, List

from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, seg_to_text
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from data_generator.tokenizer_wo_tf import get_tokenizer


class DeletionSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn, target_idx):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        self.target_idx = target_idx

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        base_probs = self.predict_fn([NLIInput(t1, t2)])[0]
        def get_score_for(t1: SegmentedText, t2: SegmentedText):
            probs = self.predict_fn([NLIInput(t1, t2)])[0]
            return base_probs[self.target_idx] - probs[self.target_idx]

        scores1 = []
        for i1 in t1.enum_seg_idx():
            t1_sub = t1.get_dropped_text([i1])
            score = get_score_for(t1_sub, t2)
            scores1.append(score)

        scores2 = []
        for i2 in t2.enum_seg_idx():
            t2_sub = t2.get_dropped_text([i2])
            score = get_score_for(t1, t2_sub)
            scores2.append(score)
        return scores1, scores2


class DeletionSolverKeras(TokenScoringSolverIF):
    def __init__(self, predict_fn, target_idx):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        self.target_idx = target_idx

    def solve_one(self, text1_tokens: List[str], text2_tokens: List[str]) -> List[float]:
        def join_token_pairs(tokens1, tokens2):
            return " ".join(tokens1), " ".join(tokens2)

        base_pair = join_token_pairs(text1_tokens, text2_tokens)
        payload: List[Tuple[str, str]] = []
        payload.append(base_pair)
        for i in range(len(text2_tokens)):
            new_tokens2 = text2_tokens[i:] + text2_tokens[i+1:]
            payload.append(join_token_pairs(text1_tokens, new_tokens2))

        probs = self.predict_fn(payload)
        probs_d = dict(zip(payload, probs))
        base_prob = probs_d[base_pair]
        scores = []
        for i in range(len(text2_tokens)):
            new_tokens2 = text2_tokens[i:] + text2_tokens[i+1:]
            case_input = join_token_pairs(text1_tokens, new_tokens2)
            case_prob = probs_d[case_input]
            score = base_prob[self.target_idx] - case_prob[self.target_idx]
            scores.append(score)
        return scores

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        s2 = self.solve_one(text1_tokens, text2_tokens)
        s1 = self.solve_one(text2_tokens, text1_tokens)
        return s1, s2
