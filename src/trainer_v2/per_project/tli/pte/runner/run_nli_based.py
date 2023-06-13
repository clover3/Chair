from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
import logging
import sys
from dataset_specific.scientsbank.eval_helper import solve_eval_report
from dataset_specific.scientsbank.parse_fns import get_split_spec
from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver
from trainer_v2.per_project.tli.pte.solver_loader import get_solver_by_name


def main():
    name = sys.argv[1]
    split_name = sys.argv[2]
    split = get_split_spec(split_name)
    run_name = f"pte_{name}_{split_name}"
    with JobContext(run_name):
        c_log.setLevel(logging.DEBUG)
        c_log.info("Building solver")
        solver = get_solver_by_name(name)
        c_log.info("Applying solver")
        eval_res = solve_eval_report(solver, split)


if __name__ == "__main__":
    main()
