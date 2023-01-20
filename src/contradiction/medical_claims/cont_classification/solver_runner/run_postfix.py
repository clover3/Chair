from contradiction.medical_claims.cont_classification.run_eval_fns import run_cont_prob_solver_and_save
from contradiction.medical_claims.cont_classification.solvers.postfix_solver import get_post_fix_idf_ignore


def main():
    for split in ["dev"]:
        solver = get_post_fix_idf_ignore(split)
        run_name = "nli_postfix_ignore"
        run_cont_prob_solver_and_save(solver, run_name, split)


if __name__ == "__main__":
    main()