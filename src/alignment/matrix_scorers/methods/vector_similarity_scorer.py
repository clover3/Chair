from typing import List, Callable

import numpy as np

from alignment import MatrixScorerIF
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from bert_api import SegmentedInstance


class VectorSimilarityScorer(MatrixScorerIF):
    def __init__(self,
                 get_vector: Callable[[List[int]], np.array],
                 get_similarity
                 ):
        self.get_vector = get_vector
        self.get_similarity = get_similarity
        
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()
        output_table = []
        score_for_not_found = 0.14
        for i1 in range(l1):
            tokens1: List[int] = inst.text1.get_tokens_for_seg(i1)
            try:
                scores_per_seg: List[float] = []
                v1 = self.get_vector(tokens1)
                for i2 in range(l2):
                    tokens2: List[int] = inst.text2.get_tokens_for_seg(i2)
                    try:
                        v2 = self.get_vector(tokens2)
                        score = self.get_similarity(v1, v2)
                    except KeyError:
                        score = score_for_not_found
                    scores_per_seg.append(score)
            except KeyError:
                scores_per_seg: List[float] = [score_for_not_found] * inst.text2.get_seg_len()

            output_table.append(scores_per_seg)
        return ContributionSummary(output_table)


# Use exact match when OOV
class VectorSimilarityScorer2(MatrixScorerIF):
    def __init__(self,
                 get_vector: Callable[[List[int]], np.array],
                 get_similarity
                 ):
        self._get_vector = get_vector
        self.get_similarity = get_similarity

    def get_vector(self, w):
        try:
            return self._get_vector(w)
        except KeyError:
            return None

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()
        output_table = []
        score_for_not_found = 0.14
        for i1 in range(l1):
            tokens1: List[int] = inst.text1.get_tokens_for_seg(i1)
            scores_per_seg: List[float] = []
            v1 = self.get_vector(tokens1)
            for i2 in range(l2):
                tokens2: List[int] = inst.text2.get_tokens_for_seg(i2)
                v2 = self.get_vector(tokens2)
                score = self.get_similarity(v1, v2)
                scores_per_seg.append(score)

            output_table.append(scores_per_seg)
        return ContributionSummary(output_table)

