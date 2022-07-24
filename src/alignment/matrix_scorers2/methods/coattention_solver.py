
from typing import List

import numpy as np

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text, SegmentedText, \
    merge_subtoken_level_scores_2d
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.attention_extractor import AttentionExtractor
from misc_lib import average


class CoAttentionSolver(MatrixScorerIF2):
    def __init__(self, attention_predictor):
        self.predictor = attention_predictor
        self.tokenizer = get_tokenizer()

    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, tokens1)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, tokens2)
        l1 = len(t1.tokens_ids)
        l2 = len(t2.tokens_ids)
        attn_score1 = self._solve_one_way(t1, t2)  # [l1, l2]
        attn_score2 = self._solve_one_way(t2, t1)  # [l2, l1]

        assert attn_score1.shape[0] == l1
        assert attn_score1.shape[1] == l2
        attn_score2_t = np.transpose(attn_score2, [1, 0])
        attn_mean = (attn_score1 + attn_score2_t) / 2  # [l1, l2]
        return merge_subtoken_level_scores_2d(average, attn_mean, t1, t2)

    def _solve_one_way(self, t1: SegmentedText, t2: SegmentedText) -> np.array:
        # [n_layer, n_head, max_seq_length, max_seq_length]
        attention_scores = self.predictor(t1.tokens_ids, t2.tokens_ids)
        attn_score = np.mean(attention_scores, axis=0)
        attn_score = np.mean(attn_score, axis=0)  # [max_seq_length, max_seq_length]

        l1 = len(t1.tokens_ids)
        l2 = len(t2.tokens_ids)
        t1_st = 1
        t1_ed = t1_st + l1
        t2_st = t1_ed + 1
        t2_ed = t2_st + l2

        attn_in = attn_score[t1_st:t1_ed, t2_st:t2_ed]  # shape = [l1, l2]
        attn_out = attn_score[t2_st:t2_ed, t1_st:t1_ed]  # shape = [l2, l1]
        attn_out = np.transpose(attn_out, [1, 0])
        attn_mean = (attn_in+attn_out) / 2
        return attn_mean


def get_co_attention_solver():
    ae = AttentionExtractor()
    return CoAttentionSolver(ae.predict)

