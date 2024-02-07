import sys

from list_lib import left
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import pearson_r_wrap


def main():

    def load_table(table_path):
        itr = tsv_iter(table_path)
        ret = []
        for qt, dt, score_s in itr:
            row = (qt, dt), float(score_s)
            ret.append(row)

        return ret

    # Example usage
    series1 = load_table(sys.argv[1])
    series1_d = dict(series1)
    series2 = load_table(sys.argv[2])
    series2_d = dict(series2)

    keys = left(series1)
    scores1 = [series1_d[key] for key in keys]
    scores2 = [series2_d[key] for key in keys]
    print(pearson_r_wrap(scores1, scores2))


if __name__ == "__main__":
    main()