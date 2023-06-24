from contradiction.medical_claims.token_tagging.acc_eval.show_score import compute_binary_metrics_for_test
from contradiction.medical_claims.token_tagging.print_score.print_MAP import BioClaimMapCalc
from tab_print import print_table


def main():
    run_list = ["random",
                "exact_match", "word2vec_em",
                "senli",
                "coattention", "lime", "deletion", "slr",
                "word_seg", "nlits87", "davinci", "gpt-3.5-turbo"
                ]
    column_list = [
        # ("map", "mismatch"),
        ("f1", "mismatch"),
        ("accuracy", "mismatch"),
        ("f1", "conflict"),
        ("accuracy", "conflict"),
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
    main()