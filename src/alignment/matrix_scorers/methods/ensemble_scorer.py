from bert_api import SegmentedInstance
from alignment import MatrixScorerIF
from typing import List
import numpy as np

from alignment.data_structure.matrix_scorer_if import ContributionSummary


class EnsembleScorer(MatrixScorerIF):
    def __init__(self, solver_list: List[MatrixScorerIF], weight_list: List[float] = None):
        self.scorer_list = solver_list
        if weight_list is not None:
            self.weight_list = weight_list
        else:
            self.weight_list = np.ones([len(solver_list)]) * 1 / len(solver_list)

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        scores_list = []
        for solver in self.scorer_list:
            score_table = solver.eval_contribution(inst)
            scores_list.append(np.array(score_table.table))

        def combine(scores_list) -> List[List[float]]:
            stacked = np.stack(scores_list, axis=0)
            weights = np.expand_dims(self.weight_list, 1)
            weights = np.expand_dims(weights, 1)
            assert len(np.shape(stacked)) == 3
            return np.sum(stacked * weights, axis=0).tolist()

        combined_scores: List[List[float]] = combine(scores_list)
        return ContributionSummary(combined_scores)

