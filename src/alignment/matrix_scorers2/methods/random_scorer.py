import random
from typing import List

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2


class RandomOne(MatrixScorerIF2):
    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        l1 = len(tokens1)
        l2 = len(tokens2)

        output_table = []
        for i1 in range(l1):
            scores_per_seg: List[float] = [0 for _ in range(l2)]
            i2 = random.randint(0, l2-1)
            scores_per_seg[i2] = 1
            output_table.append(scores_per_seg)
        return output_table