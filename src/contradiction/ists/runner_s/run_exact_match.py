from contradiction.ists.predict_common import eval_ists_noali_headlines_train
from contradiction.medical_claims.token_tagging.solvers.exact_match_solver import ExactMatchSolver
from misc_lib import tprint


def main():
    run_name = "exact_match"
    tprint("Building solver")
    solver = ExactMatchSolver()
    tprint("Building solver DONE")
    eval_ists_noali_headlines_train(run_name, solver)


if __name__ == "__main__":
    main()
