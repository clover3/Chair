from tab_print import print_table
from trainer_v2.per_project.transparency.mmp.bias.exp2.pairwise_ttest import analyze_pairwise_ttest_car


def main():
    method_list = [
        "contriever",
        "contriever-msmarco",
        "ce_mini_lm",
        "splade",
    ]

    table = []
    for method in method_list:
        score_log_path = f"output/mmp/bias/car_exp/exp3/{method}.tsv"
        yes_cnt, no_cnt = analyze_pairwise_ttest_car(score_log_path)
        row = [method, yes_cnt, no_cnt]
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()