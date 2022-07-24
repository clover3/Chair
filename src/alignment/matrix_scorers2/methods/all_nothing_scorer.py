from typing import List

import numpy as np

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2


class AllOneScorer2(MatrixScorerIF2):
    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        l1 = len(tokens1)
        l2 = len(tokens2)
        scores = np.ones([l1, l2], np.int)
        return scores.tolist()


class AllZeroScorer2(MatrixScorerIF2):
    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        l1 = len(tokens1)
        l2 = len(tokens2)
        scores = np.zeros([l1, l2], np.int)
        return scores.tolist()



