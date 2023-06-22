from trainer_v2.chair_logging import c_log
import logging
import sys
from dataset_specific.scientsbank.eval_helper import solve_eval_report
from dataset_specific.scientsbank.parse_fns import get_split_spec, sci_ents_test_split_list
from trainer_v2.per_project.tli.pte.slr_solver import SLRSolverForPTE, read_scores_for_split


def main():
    name = 'slr'
    todo = ["train_sub"] + sci_ents_test_split_list
    for split_name in todo:
        split = get_split_spec(split_name)
        token_score_d = read_scores_for_split(split)
        c_log.info("Building solver")
        solver = SLRSolverForPTE(token_score_d, name)
        c_log.info("Applying solver")
        eval_res = solve_eval_report(solver, split)


if __name__ == "__main__":
    main()
