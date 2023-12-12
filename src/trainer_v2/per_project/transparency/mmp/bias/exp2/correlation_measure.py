import sys

from scipy.stats import pearsonr, spearmanr

from cpath import output_path
from misc_lib import path_join
from table_lib import tsv_iter


def main():
    table_path = path_join(output_path, "mmp", "bias", "car_exp", "car_words_scores.tsv")
    table_path = path_join(output_path, "mmp", "car_pairs", "0.txt")

    # print("using", table_path)
    table_path = sys.argv[1]
    table_scores = tsv_iter(table_path)
    win_rates = tsv_iter(sys.argv[2])

    table_scores_d = {t.lower(): float(s) for _car, t, s in table_scores}
    win_rates_d = {t: float(s) for t, s in win_rates}
    keys = list(win_rates_d.keys())
    win_rate_l = [win_rates_d[key] for key in keys]
    table_rate_l = [table_scores_d[key] for key in keys]

    print(pearsonr(win_rate_l, table_rate_l))
    print(spearmanr(win_rate_l, table_rate_l))


if __name__ == "__main__":
    main()