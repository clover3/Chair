from typing import List

from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2


class ExactMatchScorer2(MatrixScorerIF2):
    def solve(self, tokens1: List[str], tokens2: List[str]) -> List[List[float]]:
        l1 = len(tokens1)
        l2 = len(tokens2)

        def normalize(s):
            return s.lower()

        output_table = []
        for i1 in range(l1):
            scores_per_seg: List[float] = []
            for i2 in range(l2):
                t1 = normalize(tokens1[i1])
                t2 = normalize(tokens2[i2])
                score = 1 if t1 == t2 else 0
                scores_per_seg.append(score)
            output_table.append(scores_per_seg)
        return output_table