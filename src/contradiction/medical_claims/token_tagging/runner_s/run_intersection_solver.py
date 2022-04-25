from typing import List

from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client
from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.intersection_solver import IntersectionSolver


def main():
    tag_type = "mismatch"
    run_name = "intersection"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    cache_client = get_nli_cache_client("localhost")
    solver = IntersectionSolver(cache_client.predict, False)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()