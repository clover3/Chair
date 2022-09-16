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
