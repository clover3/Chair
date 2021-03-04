import os
import sys

from scipy import stats

from list_lib import lmap
from misc_lib import average


def get_score_per_file(input_path):
    scores = []
    for line in open(input_path, "r"):
        scores.append(float(line))
    return scores

items = ["robust_3B", "robust_3A",
"robust_3I2",
"robust_3AC",
"robust_3U",
"robust_3S",
"robust_3Q",
"robust_3T",]

def main():
    run1 = sys.argv[1]
    run2 = sys.argv[2]

    stat_test_wrap(run1, run2)
    # print("baseline:", average(l1))
    # print("treatment:", average(l2))
    # print(d, p_value)


def stat_test_wrap(run1, run2):
    common_dir = os.path.join('work', "ndcg")
    def load_scores(run_name):
        scores_list = []
        for i in range(5):
            file_name = "{}_combine_{}.txt".format(run_name, i)
            scores = get_score_per_file(os.path.join(common_dir, file_name))
            scores_list.append(scores)
        return scores_list

    def select_median(scores_list):
        scores_list.sort(key=average)
        mid = int(len(scores_list) / 2)
        return scores_list[mid]

    def get_rep_score(run_name):
        return select_median(load_scores(run_name))

    l1 = get_rep_score(run1)
    l2 = get_rep_score(run2)
    assert len(l1) == len(l2)
    return stats.ttest_rel(l1, l2)


def main2():
    n_item = len(items)
    for i in range(n_item):
        for j in range(n_item):
            d, p_value = stat_test_wrap(items[i], items[j])
            if d < 0:
                if p_value < 0.01:
                    s = "0.01"
                elif p_value < 0.05:###
                    s = "0.05"
                else:
                    s = ""

                if s:
                    print(items[i], items[j], s)


if __name__ == "__main__":
    main2()