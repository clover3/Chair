import logging
from typing import List, Tuple

from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text
from contradiction.medical_claims.token_tagging.intersection_search.dev_align import get_scores_by_many_perturbations
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.chair_logging import c_log


class SearchSolver(TokenScoringSolverIF):
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
        self.tokenizer = get_tokenizer()
        c_log.setLevel(logging.WARN)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)

        scores2 = get_scores_by_many_perturbations(self.predict_fn, t1, t2)
        scores1 = get_scores_by_many_perturbations(self.predict_fn, t2, t1)
        return scores1, scores2