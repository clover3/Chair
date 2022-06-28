import os
from typing import List

from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.medical_claims.token_tagging.batch_solver_common import make_ranked_list_w_batch_solver
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from data_generator.NLI.enlidef import NEUTRAL


def run_nlits_solver(run_name, tag_type, target_label: int):
    solver = get_batch_solver_nlits(run_name, "two_seg", target_label)
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    print("Using {} problems".format(len(problems)))
    make_ranked_list_w_batch_solver(problems, run_name, save_path, tag_type, solver)


def main():
    tag_type = "mismatch"
    target_label = NEUTRAL
    run_name = "nlits45"
    run_nlits_solver(run_name, tag_type, target_label)


if __name__ == "__main__":
    main()
