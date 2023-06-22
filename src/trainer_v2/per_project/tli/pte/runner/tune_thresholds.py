from dataset_specific.scientsbank.parse_fns import SplitSpec, get_split_spec

import sys
from typing import List

from dataset_specific.scientsbank.eval_fns import evaluate
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, SplitSpec
from dataset_specific.scientsbank.pte_data_types import Question, PTEPredictionPerQuestion
from dataset_specific.scientsbank.save_load_pred import load_pte_preds_from_file
from misc_lib import get_second
from tab_print import tab_print_dict
from trainer_v2.per_project.tli.pte.path_helper import get_score_save_path, get_threshold_save_path


def change_preds(preds: List[PTEPredictionPerQuestion], t):
    for per_q in preds:
        for per_sa in per_q.per_student_answer_list:
            for per_fa in per_sa.facet_pred:
                per_fa.pred = per_fa.score > t


def save_threshold(name, value):
    open(get_threshold_save_path(name), "w").write(str(value))


def main():
    tune_split = "train_sub"
    split = get_split_spec(tune_split)
    questions: List[Question] = load_scientsbank_split(split.split_name)
    solver_name_list = ["em", "w2v", "coattention", "lime", "deletion", "senli", "nli14", "nlits"]
    solver_name_list = ["slr"]
    def get_t_list(solver_name):
        if solver_name == "coattention":
            return [t * 0.002 for t in range(0, 51)]
        elif solver_name in ["deletion", "senli", "lime", 'slr']:
            return [t * 0.1 for t in range(-10, 20)]
        else:
            return [t * 0.02 for t in range(0, 51)]

    for solver_name in solver_name_list:
        records = []
        run_name = f"{solver_name}_{split.get_save_name()}"
        save_path = get_score_save_path(run_name)
        preds = load_pte_preds_from_file(save_path)

        for t in get_t_list(solver_name):
            change_preds(preds, t)
            eval_res = evaluate(questions, preds)
            score = eval_res['macro_f1']
            records.append((t, score))

        for row in records:
            print(row)
        records.sort(key=get_second, reverse=True)
        print(solver_name, records[0])
        t, score = records[0]
        save_threshold(solver_name, t)



if __name__ == "__main__":
    main()