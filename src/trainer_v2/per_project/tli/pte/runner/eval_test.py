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
        "em", "w2v", "coattention", "lime", "deletion",
        "senli", "nli14", "nlits", 'all_true', 'gpt-3.5-turbo']
    no_tune = ['all_true', 'gpt-3.5-turbo']

    split_list = sci_ents_test_split_list
    head = ["run_name"] + split_list
    n_item_per_split = {
        'train_first': 167,
        'train_sub': 1706,
        'test-unseen-answers': 1831,
        'test-unseen-domains': 15966,
        'test-unseen-questions': 2693
    }
    table = [head]
    for solver_name in solver_name_list:
        row = [solver_name]
        for split_name in split_list:
            split = get_split_spec(split_name)
            run_name = f"{solver_name}_{split.get_save_name()}"
            if solver_name in no_tune:
                pass
            else:
                run_name = run_name + "_t"
            questions: List[Question] = load_scientsbank_split(split)
            try:
                save_path = get_score_save_path(run_name)
                preds = load_pte_preds_from_file(save_path)
                if len(preds) != len(questions):
                    c_log.warning("Run %s has %d entries while dataset has %d",
                                  run_name, len(preds), len(questions))
                eval_res = evaluate(questions, preds)
                score = eval_res['macro_f1']
                assert eval_res['total'] == n_item_per_split[split_name]
            except FileNotFoundError:
                pass
                score = "-"
            row.append(score)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()
