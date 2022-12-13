

from contradiction.medical_claims.cont_classification.run_eval_fns import do_eval, eval_auc
from tab_print import print_table


def main():
    split = "dev"
    run_name_list = [
        "nli", "nli_q", "nli_q3", "nli_q4",
        "tnli1", "tnli3",
        "tnli4", "tnli5"
    ]

    metric_names = ["auc", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]

    table = []
    table.append(["run_name"] + metric_names)
    for run_name in run_name_list:
        scores = do_eval(run_name, split)
        scores['auc'] = float(eval_auc(run_name, split))
        row = [run_name]
        for metric in metric_names:
            s = scores[metric]
            if type(s) == float:
                s = "{0:.3f}".format(s)
            row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()