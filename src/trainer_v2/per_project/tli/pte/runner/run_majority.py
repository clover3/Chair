import logging

from dataset_specific.scientsbank.eval_helper import solve_eval_report, report_macro_f1
from dataset_specific.scientsbank.parse_fns import SplitSpec
from dataset_specific.scientsbank.pte_solver_if import PTESolverAllTrue
from tab_print import tab_print_dict
from trainer_v2.chair_logging import c_log

from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver


def main():
    c_log.setLevel(logging.DEBUG)
    split = SplitSpec("train",  use_subset=True, subset_portion=0.1)
    c_log.info("Building solver")
    solver = PTESolverAllTrue()
    c_log.info("Applying solver")
    eval_res = solve_eval_report(solver, split)
    tab_print_dict(eval_res)

    name = solver.get_name()
    report_macro_f1(eval_res, name, split)


if __name__ == "__main__":
    main()
