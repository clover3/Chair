import logging

from contradiction.medical_claims.cont_classification.run_eval_fns import run_cont_prob_solver_and_save
from contradiction.medical_claims.cont_classification.solvers.token_nli import get_token_level_inf_classifier
from trainer_v2.chair_logging import c_log


def main():
    for split in ["dev", "test"]:
        run_name = "tnli3"
        solver = get_token_level_inf_classifier(run_name)
        run_cont_prob_solver_and_save(solver, run_name, split)
        break


if __name__ == "__main__":
    main()