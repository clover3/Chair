import os

from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.gpt_solver.get_instruct_gpt_solver import get_gpt_solver_mismatch, \
    get_gpt_requester_mismatch, get_gpt_file_solver_mismatch
from contradiction.medical_claims.token_tagging.util import solve_and_save


def main():
    tag_type = "mismatch"
    run_name = "davinci"
    solver = get_gpt_file_solver_mismatch()
    solve_and_save(run_name, solver, tag_type)


if __name__ == "__main__":
    main()
