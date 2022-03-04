import random
from typing import List, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF, \
    make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem


class RandomScorer(TokenScoringSolverIF):
    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = [random.random() for _ in text1_tokens]
        scores2 = [random.random() for _ in text2_tokens]
        return scores1, scores2


def main():
    tag_type = "conflict"
    run_name = "random"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, RandomScorer())


if __name__ == "__main__":
    main()