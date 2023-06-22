from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import get_tf_idf_solver
from contradiction.medical_claims.token_tagging.util import solve_and_save
from contradiction.medical_claims.token_tagging.solvers.idf_solver import ConvolutionSolver
from misc_lib import tprint


def run_tf_idf_conv():
    tag_type = "mismatch"
    run_name = "tfidf-conv"
    mask = [0.1, 0.2, 0.5, 0.2, 0.1]
    solver = ConvolutionSolver(get_tf_idf_solver(), mask)
    tprint("running solver")
    solve_and_save(run_name, solver, tag_type)
    do_ecc_eval_w_trec_eval(run_name, tag_type)



if __name__ == "__main__":
    run_tf_idf_conv()
