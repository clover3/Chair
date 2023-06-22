from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from misc_lib import TEL


def solve_and_save(run_name, solver, tag_type):
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def apply_solver(problems: List[AlamriProblem],
                 solver):
    def problem_to_scores(p: AlamriProblem):
        return solver.solve_from_text(p.text1, p.text2)

    for p in TEL(problems):
        scores1, scores2 = problem_to_scores(p)