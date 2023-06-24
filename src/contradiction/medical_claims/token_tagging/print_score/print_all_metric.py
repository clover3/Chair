from contradiction.medical_claims.token_tagging.acc_eval.show_score import compute_binary_metrics_for_test
from contradiction.medical_claims.token_tagging.path_helper import mismatch_only_method_list
from contradiction.medical_claims.token_tagging.print_score.print_MAP import BioClaimMapCalc
from tab_print import print_table


def compute_map(run_name, tag):
    split = "test"
    scorer = BioClaimMapCalc(split)
    return scorer.compute(run_name, tag)


def show_for_mismatch():
    run_list = ["random",
                "exact_match",
                "word2vec_em",
                "coattention",
                "lime",
                "deletion",
                "senli",
                "slr",
                "word_seg",
                "nlits87",
                "davinci",
                "gpt-3.5-turbo"
                ]

    metric_list = ["precision", "recall", "f1", "accuracy", "map"]
    tag_list = ["mismatch", "conflict"]
    head = ["run name"]
    head += metric_list
    head += metric_list
    table = [head]

    for run_name in run_list:
        row = [run_name]
        for tag in tag_list:
            for metric in metric_list:
                if metric == "map":
                    s = compute_map(run_name, tag)
                else:
                    if metric == "accuracy":
                        metric_tuned_for = metric
                    else:
                        metric_tuned_for = "f1"
                    score_d = compute_binary_metrics_for_test(run_name, tag, metric_tuned_for)
                    if metric in score_d:
                        s = score_d[metric]
                    elif metric_tuned_for in score_d:
                        s = "-"
                    else:
                        raise KeyError

                if tag == "conflict" and run_name in mismatch_only_method_list:
                    s = "-"

                row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    show_for_mismatch()