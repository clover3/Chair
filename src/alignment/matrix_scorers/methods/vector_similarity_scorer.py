from bert_api import SegmentedInstance
from alignment import MatrixScorerIF
from typing import List
import numpy as np
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from typing import List, Iterable, Callable, Dict, Tuple, Set


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
        for i1 in range(l1):
            scores_per_seg: List[float] = []
            tokens1: List[int] = inst.text1.get_tokens_for_seg(i1)
            v1 = self.get_vector(tokens1)
            for i2 in range(l2):
                tokens2: List[int] = inst.text2.get_tokens_for_seg(i2)
                v2 = self.get_vector(tokens2)
                scores_per_seg.append(self.get_similarity(v1, v2))
            output_table.append(scores_per_seg)

        return ContributionSummary(output_table)

