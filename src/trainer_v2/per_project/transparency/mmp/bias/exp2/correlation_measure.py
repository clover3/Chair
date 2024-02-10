import sys

from scipy.stats import pearsonr, spearmanr

from cpath import output_path
from misc_lib import path_join
from table_lib import tsv_iter


def main():
    table_path = sys.argv[1]
    table_scores = tsv_iter(table_path)
    win_rates = tsv_iter(sys.argv[2])

    q_term = "car"

    def parse_table_scores(table_scores):
        out_d = {}
        for row in table_scores:
            if len(row) == 2:
                term, score = row
                out_d[term.lower()] = float(score)

            elif len(row) == 3 :
                q_term_, term, score = row
                if q_term_ == q_term:
                    out_d[term.lower()] = float(score)

            else:
                raise ValueError()
        return out_d

    table_scores_d = parse_table_scores(table_scores)
    print(f"Loaded {len(table_scores_d)} entries for term={q_term}")
    win_rates_d = {t: float(s) for t, s in win_rates}
    common_keys = list(win_rates_d.keys())
    common_keys = [k for k in common_keys if k in table_scores_d]

    if len(common_keys) != len(win_rates_d):
        print("Only {} of {} terms are found".format(len(common_keys), len(win_rates_d)))

    win_rate_l = [win_rates_d[key] for key in common_keys]
    table_rate_l = [table_scores_d[key] for key in common_keys]

    print(pearsonr(win_rate_l, table_rate_l))


if __name__ == "__main__":
    main()