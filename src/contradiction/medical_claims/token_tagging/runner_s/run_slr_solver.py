import sys
from cpath import output_path
from misc_lib import path_join

from contradiction.medical_claims.token_tagging.batch_solver_common import make_ranked_list_w_batch_solver
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem
from contradiction.medical_claims.token_tagging.solvers.slr_baseline import get_snli_logic_solver


def main():
    problems = load_alamri_problem()
    run_name = "slr"
    for tag_type in ["mismatch", "conflict"]:
        score_path = path_join(output_path, "alamri_annotation1", "slr", f"slr_{tag_type}.txt")
        solver = get_snli_logic_solver(problems, score_path)
        save_path = get_save_path2(run_name, tag_type)
        make_ranked_list_w_batch_solver(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()