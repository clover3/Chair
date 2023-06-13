import sys
from typing import List

from dataset_specific.scientsbank.eval_fns import evaluate
from dataset_specific.scientsbank.eval_helper import report_macro_f1
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, SplitSpec
from dataset_specific.scientsbank.pte_data_types import Question
from dataset_specific.scientsbank.save_load_pred import load_pte_preds_from_file
from tab_print import tab_print_dict
from trainer_v2.per_project.tli.pte.path_helper import get_score_save_path


def main():
    solver_name = sys.argv[1]
    split = SplitSpec("train", True, 0.1)
    split_name = split.get_save_name()
    questions: List[Question] = load_scientsbank_split(split)

    run_name = f"{solver_name}_{split_name}"
    save_path = get_score_save_path(run_name)
    preds = load_pte_preds_from_file(save_path)
    eval_res = evaluate(questions, preds)

    tab_print_dict(eval_res)
    report_macro_f1(eval_res, solver_name, split)


if __name__ == "__main__":
    main()
