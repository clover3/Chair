from typing import List, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF


def overlap_check_1_if_no(text1_tokens, text2_tokens) -> List[int]:
    scores = []
    for t1 in text1_tokens:
        if t1 in text2_tokens:
            scores.append(0)
        else:
            scores.append(1)
    return scores


# Assign 1 if no exact match exists
class ExactMatchSolver(TokenScoringSolverIF):
    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = overlap_check_1_if_no(text1_tokens, text2_tokens)
        scores2 = overlap_check_1_if_no(text2_tokens, text1_tokens)
        return scores1, scores2
