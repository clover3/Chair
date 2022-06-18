from typing import List

from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.keyword_solver import KeywordSolver
from misc_lib import tprint


def main4():
    tag_type = "conflict"
    run_name = "keyword_not"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    tprint("Building solver")
    solver = KeywordSolver(["not"])
    tprint("Building solver DONE")
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)
    do_ecc_eval_w_trec_eval(run_name, tag_type)


if __name__ == "__main__":
    main4()
