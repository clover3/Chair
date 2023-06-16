import os

from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.gpt_solver.get_instruct_gpt_solver import get_gpt_solver_mismatch, \
    get_gpt_requester_mismatch
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import solve_and_save


def main():
    tag_type = "mismatch"
    run_name = "davinci_mismatch"
    solver = get_gpt_solver_mismatch()
    solve_and_save(run_name, solver, tag_type)
    do_ecc_eval_w_trec_eval(run_name, tag_type)


def main():
    tag_type = "mismatch"
    run_name = "davinci_mismatch_dummy"
    solver = get_gpt_requester_mismatch()
    solve_and_save(run_name, solver, tag_type)


if __name__ == "__main__":
    main()
