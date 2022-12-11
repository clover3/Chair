
from contradiction.medical_claims.cont_classification.run_eval_fns import do_eval, eval_per_category
from tab_print import print_table


def main():
    split = "dev"
    run_name_list = [
        "majority", "random", "bm25_tuned", "nli", "nli_q2"
    ]

    metric_names = ["accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    table = []
    table.append(["run_name"] + metric_names)
    for run_name in run_name_list:
        scores_d = eval_per_category(run_name, split)
        for key, scores in scores_d.items():
            run_name_comb = run_name + "_" + key
            row = [run_name_comb]
            for metric in metric_names:
                s = scores[metric]
                if type(s) == float:
                    s = "{0:.3f}".format(s)
                row.append(s)
            table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()
