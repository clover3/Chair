from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.probe_solver import ProbeSolver
from data_generator.NLI.enlidef import NEUTRAL, CONTRADICTION
from explain.bert_components.cls_probe_predictor import CacheClsProbePredictor


def main():
    tag_type = "mismatch"
    run_name = "probe"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    solver = ProbeSolver(CacheClsProbePredictor(), NEUTRAL)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def solve_conflict():
    tag_type = "conflict"
    run_name = "probe"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    solver = ProbeSolver(CacheClsProbePredictor(), CONTRADICTION)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()
