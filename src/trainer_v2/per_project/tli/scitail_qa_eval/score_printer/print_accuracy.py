from evals.basic_func import get_acc_prec_recall_i
from misc_lib import two_digit_float
from tab_print import print_table
from trainer_v2.per_project.tli.scitail_qa_eval.eval_helper import load_scores, load_scitail_qa_label
from trainer_v2.per_project.tli.scitail_qa_eval.path_helper import get_score_save_path
from typing import List, Iterable, Callable, Dict, Tuple, Set


def apply_threshold(scores, threshod):
    output = []
    for s in scores:
        if s >= threshod:
            output.append(1)
        else:
            output.append(0)
    return output


def main():
    split = "dev"
    run_name_list = [
        "bm25_clue",
        "bm25_tuned",
        "nli_rev_direct",
        "scitail_rev_direct",
        "nli_pep",
        "nli_pep_idf",
        "tnli2"
    ]
    column_head = ["run_name", "accuracy", "precision", "recall", "f1",]
    table = [column_head]
    labels = load_scitail_qa_label(split)
    for run_name in run_name_list:
        save_name = f"{run_name}_{split}"
        save_path = get_score_save_path(save_name)
        scores: List[float] = load_scores(save_path)
        pred = apply_threshold(scores, 0.5)
        d = get_acc_prec_recall_i(pred, labels)

        row = [run_name]
        for metric in column_head[1:]:
            row.append(two_digit_float(d[metric]))

        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()
