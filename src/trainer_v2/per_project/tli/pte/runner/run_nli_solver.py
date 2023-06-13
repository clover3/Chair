import logging
import sys

from dataset_specific.scientsbank.eval_helper import solve_eval_report
from dataset_specific.scientsbank.parse_fns import SplitSpec, get_split_spec
from dataset_specific.scientsbank.pte_solver_if import PTESolverIF
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_predictor, get_nli14_direct
from trainer_v2.per_project.config_util import get_strategy_with_default_pu_config
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_1
from trainer_v2.per_project.tli.pte.nlits_solver import PTESolverTLI


def main():
    name = "token-entail"
    split_name = sys.argv[1]
    split = get_split_spec(split_name)
    run_name = f"pte_{name}_{split_name}"
    with JobContext(run_name):
        strategy = get_strategy_with_default_pu_config()
        c_log.info("Building solver")

        nli_predict_fn = get_nli14_direct(strategy)
        combine_tli = lambda x: x[:, 0]
        solver = PTESolverTLI(nli_predict_fn, enum_subseq_1, combine_tli, "nli14")

        c_log.info("Applying solver")
        solve_eval_report(solver, split)


if __name__ == "__main__":
    main()
