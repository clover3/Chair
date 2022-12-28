from contradiction.medical_claims.token_tagging.acc_eval.show_score import compute_binary_metrics_for_test
from contradiction.medical_claims.token_tagging.print_score.print_MAP import BioClaimMapCalc
from tab_print import print_table


def show_for_mismatch():
    run_list = ["random", "exact_match", "word2vec_em",
                "coattention", "lime",
                "word_seg", "nlits87", "davinci"
                ]

    column_list = [
        ("map", "mismatch"),
        ("accuracy", "mismatch"),
        ("f1", "mismatch"),
        ("map", "conflict"),
        ("accuracy", "conflict"),
        ("f1", "conflict"),
    ]
    split = "test"
    scorer = BioClaimMapCalc(split)
    head = ["run name"]
    head.extend(map(" ".join, column_list))
    table = [head]

    for run_name in run_list:
        row = [run_name]
        for metric, tag in column_list:
            if metric == "map":
                s = scorer.compute(run_name, tag)
            else:
                score_d = compute_binary_metrics_for_test(run_name, tag, metric)
                s = score_d[metric]
            row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    show_for_mismatch()