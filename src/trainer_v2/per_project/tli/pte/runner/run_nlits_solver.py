import sys

from dataset_specific.scientsbank.eval_helper import solve_eval_report
from dataset_specific.scientsbank.parse_fns import get_split_spec
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.tli.pte.solver_loader import get_nlits_pte_solver


def main():
    split_name = sys.argv[1]
    split = get_split_spec(split_name)
    name = "nlits"
    run_name = f"pte_{name}_{split_name}"

    with JobContext(run_name):
        c_log.info("Building solver")
        solver = get_nlits_pte_solver(name)
        c_log.info("Applying solver")
        solve_eval_report(solver, split)


if __name__ == "__main__":
    main()
