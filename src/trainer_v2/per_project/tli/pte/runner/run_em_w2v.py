import logging

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.solvers.exact_match_solver import ExactMatchSolver
from contradiction.medical_claims.token_tagging.solvers.word2vec_solver import get_word2vec_solver
from dataset_specific.scientsbank.eval_helper import solve_eval_report, report_macro_f1
from dataset_specific.scientsbank.parse_fns import SplitSpec, get_split_spec
import nltk
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
import logging
import sys

from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver


def get_solver_by_name(name) -> TokenScoringSolverIF:
    if name == "w2v":
        return get_word2vec_solver()
    elif name == "em":
        return ExactMatchSolver()


def main():
    name = sys.argv[1]
    split_name = sys.argv[2]
    split = get_split_spec(split_name)
    run_name = f"pte_{name}_{split_name}"
    with JobContext(run_name):
        token_solver = get_solver_by_name(name)
        c_log.setLevel(logging.DEBUG)
        c_log.info("Building solver")

        def tokenizer(text):
            return nltk.word_tokenize(text.lower())

        solver = PTESolverFromTokenScoringSolver(token_solver, tokenizer, True, name)
        c_log.info("Applying solver")
        eval_res = solve_eval_report(solver, split)


if __name__ == "__main__":
    main()
