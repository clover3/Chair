import sys

from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bias.exp2.correlation_measure import compute_table_correlation


def main():
    method_list = [
        "ce_mini_lm",
        "splade",
        "tas_b",
        "contriever",
        "contriever-msmarco",
    ]
    table_path = sys.argv[1]
    table_scores = list(tsv_iter(table_path))
    acc_method = "avg"
    # acc_method = "win"
    for acc_method in ["avg", "rank"]:
        print(f"Use {acc_method} to compare")

        table = []
        for method in method_list:
            score_log_path = f"output/mmp/bias/car_exp/exp3/{method}_{acc_method}.tsv"
            yes_cnt, no_cnt = compute_table_correlation(table_scores, score_log_path)
            row = [method, yes_cnt, no_cnt]
            table.append(row)

        print_table(table)


if __name__ == "__main__":
    main()