import random
from typing import List

from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client
from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.search_solver import PartialSegSolver
from data_generator.NLI.enlidef import NEUTRAL, CONTRADICTION


def solve_for_tag(run_name, tag_type, target_label):
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    predict_fn = get_nli_client("localhost")
    random.seed(0)
    solver = PartialSegSolver(predict_fn, target_label)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main():
    tag_type = "conflict"
    run_name = "psearch"
    target_label = CONTRADICTION
    solve_for_tag(run_name, tag_type, target_label)


if __name__ == "__main__":
    main()
