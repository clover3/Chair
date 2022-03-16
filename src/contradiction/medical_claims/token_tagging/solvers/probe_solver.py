from typing import List, Tuple

import numpy as np

from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText, \
    merge_subtoken_level_scores
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from data_generator.NLI.enlidef import NEUTRAL
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cls_probe_predictor import ClsProbePredictorIF
from misc_lib import average


class ProbeSolver(TokenScoringSolverIF):
    def __init__(self, probe_predictor, target_label=NEUTRAL):
        self.predictor: ClsProbePredictorIF = probe_predictor
        self.tokenizer = get_tokenizer()
        self.target_label = target_label

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        t1_scores = self.solve_for_second(t2, t1)
        t2_scores = self.solve_for_second(t1, t2)
        return t1_scores, t2_scores

    def solve_for_second(self, t1: SegmentedText, t2: SegmentedText) -> List[float]:
        probe_output = self.predictor.predict(t1.tokens_ids, t2.tokens_ids)
        #
        def get_weight(i) -> float:
            return (13 - i) / 13
        layer_probes = [get_weight(i) * probe_output.get_layer_probe(i) for i in range(1, 13)]
        weight_sum: float = sum([get_weight(i) for i in range(1, 13)])

        weighted_probes = np.sum(np.stack(layer_probes, 0), axis=0) / weight_sum
        n_scores = weighted_probes[:, self.target_label]
        seg2_scores = probe_output.slice_seg2(n_scores)
        scores = merge_subtoken_level_scores(average, seg2_scores, t2)
        return scores


