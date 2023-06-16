import logging

from dataset_specific.scientsbank.eval_helper import solve_eval_report, report_macro_f1
from dataset_specific.scientsbank.parse_fns import SplitSpec, get_split_spec
from dataset_specific.scientsbank.pte_solver_if import PTESolverAllTrue
from tab_print import tab_print_dict
from trainer_v2.chair_logging import c_log

from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver


def main():
    split_name = "train_first"
    split = get_split_spec(split_name)
    c_log.setLevel(logging.DEBUG)
    c_log.info("Building solver")
    solver = PTESolverAllTrue()
    c_log.info("Applying solver")
    eval_res = solve_eval_report(solver, split)
    tab_print_dict(eval_res)


if __name__ == "__main__":
    main()
