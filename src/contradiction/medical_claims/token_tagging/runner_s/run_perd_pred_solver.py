from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, \
    load_alamri_split
from contradiction.medical_claims.token_tagging.solvers.pert_pred_solvers import get_pert_pred_solver, \
    get_pert_pred_solver_ex1


def main_old():
    tag_type = "mismatch"
    run_name = "pert_pred"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_split("dev")
    solver = get_pert_pred_solver()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main():
    tag_type = "mismatch"
    run_name = "pert_pred_ex1"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_split("dev")
    solver = get_pert_pred_solver_ex1()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()

