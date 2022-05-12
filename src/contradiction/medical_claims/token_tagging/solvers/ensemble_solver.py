from typing import List, Tuple

import numpy as np

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF


class EnsembleSolver(TokenScoringSolverIF):
    def __init__(self, solver_list: List[TokenScoringSolverIF], weight_list: List[float] = None):
        self.solver_list = solver_list
        if weight_list is not None:
            self.weight_list = weight_list
        else:
            self.weight_list = np.ones([len(solver_list)]) * 1 / len(solver_list)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores_list1 = []
        scores_list2 = []
        for solver in self.solver_list:
            scores1, scores2 = solver.solve(text1_tokens, text2_tokens)
            scores_list1.append(np.array(scores1))
            scores_list2.append(np.array(scores2))

        def combine(scores_list) -> List[float]:
            stacked = np.stack(scores_list, axis=0)
            weights = np.expand_dims(self.weight_list, 1)
            return np.sum(stacked * weights, axis=0).tolist()

        return combine(scores_list1), combine(scores_list2)


class NormalizeEnsembleScorer(TokenScoringSolverIF):
    def __init__(self, solver_list: List[TokenScoringSolverIF], normalize_range: List[float] = None):
        self.solver_list = solver_list
        if normalize_range is not None:
            self.cap_list = normalize_range
        else:
            self.cap_list = [None for _ in solver_list]

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores_list1 = []
        scores_list2 = []
        for solver in self.solver_list:
            scores1, scores2 = solver.solve(text1_tokens, text2_tokens)
            scores_list1.append(np.array(scores1))
            scores_list2.append(np.array(scores2))

        def normalize(np_array, cap):
            if cap is None:
                return np_array
            gap = max(np_array) - min(np_array)
            if gap == 0:
                gap = 1
            factor = 0.1 / gap
            return (np_array - min(np_array)) * factor

        def combine(scores_list) -> List[float]:
            new_score_list = [normalize(scores, cap) for scores, cap in zip(scores_list, self.cap_list)]
            stacked = np.stack(new_score_list, axis=0)
            return np.sum(stacked, axis=0).tolist()

        return combine(scores_list1), combine(scores_list2)

