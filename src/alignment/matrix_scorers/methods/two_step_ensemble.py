from typing import List

import numpy as np

from alignment import MatrixScorerIF
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from bert_api import SegmentedInstance


class MatNormalizeEnsembleScorer(MatrixScorerIF):
    def __init__(self, solver_list: List[MatrixScorerIF], normalize_range: List[float] = None):
        self.scorer_list = solver_list
        if normalize_range is not None:
            self.cap_list = normalize_range
        else:
            self.cap_list = [None for _ in solver_list]

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        scores_list = []
        for solver in self.scorer_list:
            score_table = solver.eval_contribution(inst)
            scores_list.append(np.array(score_table.table))

        def normalize(np_array, cap):
            if cap is None:
                return np_array
            gap = max(np_array) - min(np_array)
            if gap == 0:
                gap = 1
            factor = 0.1 / gap
            return (np_array - min(np_array)) * factor

        def combine(scores_list) -> List[List[float]]:
            new_score_list = [normalize(scores, cap) for scores, cap in zip(scores_list, self.cap_list)]
            stacked = np.stack(new_score_list, axis=0)
            return np.sum(stacked, axis=0).tolist()

        combined_scores: List[List[float]] = combine(scores_list)
        return ContributionSummary(combined_scores)

