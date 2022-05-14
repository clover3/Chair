from typing import List, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from misc_lib import validate_equal
from ptorch.ists.ists_predictor import get_ists_predictor


class ISTSSolver(TokenScoringSolverIF):
    def __init__(self):
        self.predictor = get_ists_predictor()

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.solve_one(text1_tokens, text2_tokens)
        scores2 = self.solve_one(text2_tokens, text1_tokens)
        return scores1, scores2

    def solve_one(self, text1_tokens: List[str], text2_tokens: List[str]) -> List[float]:
        probs = self.predictor.predict(text1_tokens, text2_tokens)
        l1 = len(probs)
        l2 = len(probs[0])
        max_len = max([len(text1_tokens), len(text2_tokens)])
        validate_equal(l1, max_len)
        validate_equal(l2, max_len+1)
        scores1 = []
        for idx, row in enumerate(probs):
            valid_probs = row[:len(text2_tokens)]
            s = max(valid_probs)
            scores1.append(s)
        return scores1

