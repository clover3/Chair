import sys
from typing import List

from dataset_specific.scientsbank.eval_fns import evaluate
from dataset_specific.scientsbank.eval_helper import report_macro_f1
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, SplitSpec, sci_ents_test_split_list, \
    get_split_spec
from dataset_specific.scientsbank.pte_data_types import Question
from dataset_specific.scientsbank.save_load_pred import load_pte_preds_from_file
from tab_print import tab_print_dict, print_table
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.tli.pte.path_helper import get_score_save_path


def main():
    solver_name_list = [
        "em", "w2v", "coattention", "deletion",
        "senli", "nli14", "nlits"]
    head = ["run_name"] + sci_ents_test_split_list
    table = [head]
    for solver_name in solver_name_list:
        row = [solver_name]
        for split_name in sci_ents_test_split_list:
            split = get_split_spec(split_name)
            run_name = f"{solver_name}_{split.get_save_name()}_t"
            questions: List[Question] = load_scientsbank_split(split)
            try:
                save_path = get_score_save_path(run_name)
                preds = load_pte_preds_from_file(save_path)
                if len(preds) != len(questions):
                    c_log.warning("Run %s has %d entries while dataset has %d",
                                  run_name, len(preds), len(questions))
                eval_res = evaluate(questions, preds)
                score = eval_res['macro_f1']
            except FileNotFoundError:
                pass
                score = "-"
            row.append(score)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()
