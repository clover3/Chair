import os
from typing import List
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.solvers.partial_search_solver import get_batch_partial_seg_solver
from contradiction.medical_claims.token_tagging.batch_solver_common import make_ranked_list_w_batch_solver
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem


def run_nlits_solver(run_name, tag_type):
    def sel_score_fn(probs):
        return probs[1] + probs[2]
    solver = get_batch_partial_seg_solver(sel_score_fn)
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    print("Using {} problems".format(len(problems)))
    make_ranked_list_w_batch_solver(problems, run_name, save_path, tag_type, solver)
    do_ecc_eval_w_trec_eval(run_name, tag_type)


def main():
    tag_type = "mismatch"
    run_name = "partial_seg_cn"
    run_nlits_solver(run_name, tag_type)


if __name__ == "__main__":
    main()
