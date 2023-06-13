import logging

from contradiction.medical_claims.token_tagging.solvers.lime_solver import get_lime_solver_nli14_direct
from dataset_specific.scientsbank.eval_helper import solve_eval_report, report_macro_f1
from dataset_specific.scientsbank.parse_fns import SplitSpec
from tab_print import tab_print_dict
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
import nltk

from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver


def main():
    target_idx = 0
    name = "lime"
    run_name = f"pte_{name}"
    with JobContext(run_name):
        token_solver = get_lime_solver_nli14_direct(target_idx)
        c_log.setLevel(logging.DEBUG)
        split = SplitSpec("train", use_subset=True, subset_portion=0.1)
        c_log.info("Building solver")

        def tokenizer(text):
            return nltk.word_tokenize(text.lower())

        solver = PTESolverFromTokenScoringSolver(token_solver, tokenizer, True, name)
        c_log.info("Applying solver")
        solve_eval_report(solver, split)


if __name__ == "__main__":
    main()
