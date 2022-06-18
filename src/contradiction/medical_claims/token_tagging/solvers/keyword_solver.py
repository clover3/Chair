from typing import List, Tuple

# Assign 1 if no exact match exists
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF


class KeywordSolver(TokenScoringSolverIF):
    def __init__(self, keyword_list):
        self.keyword_list = keyword_list

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.check_keywords(text1_tokens)
        scores2 = self.check_keywords(text2_tokens)
        return scores1, scores2

    def check_keywords(self, tokens) -> List[int]:
        scores = []
        for t in tokens:
            if t.lower() in self.keyword_list:
                s = 1
            else:
                s = 0
            scores.append(s)
        return scores

