import sys
from typing import List

from dataset_specific.scientsbank.eval_fns import evaluate
from dataset_specific.scientsbank.eval_helper import report_macro_f1
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, SplitSpec, sci_ents_test_split_list, \
    get_split_spec
from dataset_specific.scientsbank.pte_data_types import Question, PTEPredictionPerQuestion
from dataset_specific.scientsbank.save_load_pred import load_pte_preds_from_file
from tab_print import tab_print_dict, print_table
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.tli.pte.path_helper import get_score_save_path, get_flat_score_save_path

no_tune_method_list = ['all_true', 'all_false', 'gpt-3.5-turbo']


def main():
    solver_name_list = [
        "em", "w2v", "coattention", "lime", "deletion",
        "senli", "nli14", "nlits", 'all_true','all_false', 'slr', 'gpt-3.5-turbo']

    split_list = sci_ents_test_split_list
    for solver_name in solver_name_list:
        for split_name in split_list:
            split = get_split_spec(split_name)
            run_name = f"{solver_name}_{split.get_save_name()}"
            if solver_name in no_tune_method_list:
                pass
            else:
                run_name = run_name + "_t"
            binary_flat_pred = []
            try:
                save_path = get_score_save_path(run_name)
                preds: List[PTEPredictionPerQuestion] = load_pte_preds_from_file(save_path)
                preds.sort(key=lambda x: x.id)
                for p in preds:
                    p.per_student_answer_list.sort(key=lambda x: x.id)
                    for sa in p.per_student_answer_list:
                        sa.facet_pred.sort(key=lambda x: x.facet_id)
                        for fp in sa.facet_pred:
                            binary_flat_pred.append(int(fp.pred))
                with open(get_flat_score_save_path(run_name), "w") as f:
                    for p in binary_flat_pred:
                        f.write("{}\n".format(p))

            except FileNotFoundError:
                pass
                score = "-"


if __name__ == "__main__":
    main()