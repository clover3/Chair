from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.exact_match_solver import ExactMatchSolver


def main():
    tag_type = "mismatch"
    tag_type = "conflict"
    run_name = "exact_match"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    solver = ExactMatchSolver()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()