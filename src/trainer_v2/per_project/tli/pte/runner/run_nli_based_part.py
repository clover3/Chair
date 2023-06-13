from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
import logging
import sys
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIFOneWay
from data_generator.NLI.enlidef import ENTAILMENT, enli_tags
from dataset_specific.scientsbank.eval_helper import solve_eval_report, solve_part
from dataset_specific.scientsbank.parse_fns import get_split_spec
from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver
from trainer_v2.per_project.tli.pte.solver_loader import get_solver_by_name
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    name = sys.argv[1]
    split_name = sys.argv[2]
    job_no = int(sys.argv[3])
    split = get_split_spec(split_name)
    run_name = f"pte_{name}_{split_name}_{job_no}"
    with JobContext(run_name):
        c_log.info("Building solver")
        solver = get_solver_by_name(name)
        c_log.info("Applying solver")
        solve_part(solver, split, job_no)


if __name__ == "__main__":
    main()
