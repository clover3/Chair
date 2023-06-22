from contradiction.medical_claims.token_tagging.acc_eval.show_score import compute_binary_metrics_for_test
from contradiction.medical_claims.token_tagging.print_score.print_MAP import BioClaimMapCalc
from tab_print import print_table


def show_for_mismatch():
    run_list = ["random",
                "majority",
                "exact_match", "word2vec_em",
                "senli", "slr",
                "coattention", "lime", "deletion",
                "word_seg", "nlits87", "davinci", "gpt-3.5-turbo"
                ]
    column_list = [
        # ("map", "mismatch"),
        ("precision", "mismatch"),
        ("recall", "mismatch"),
        ("precision", "conflict"),
        ("recall", "conflict"),
    ]

    split = "test"
    head = ["run name"]
    head.extend(map(" ".join, column_list))
    table = [head]
    metric_tuned_for = "f1"

    for run_name in run_list:
        row = [run_name]
        for metric, tag in column_list:
            score_d = compute_binary_metrics_for_test(run_name, tag, metric_tuned_for)
            try:
                s = score_d[metric]
            except KeyError:
                s = "-"
            row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    show_for_mismatch()