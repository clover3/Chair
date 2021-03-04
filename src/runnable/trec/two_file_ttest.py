import sys
from typing import List, Dict

from scipy.stats import stats

from evals.metrics import get_metric_fn
from evals.parse import load_qrels_flat
from misc_lib import average
from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry


def get_score_per_query(input_path):
    scores = []
    for line in open(input_path, "r"):
        scores.append(float(line))
    return scores


def main():
    score_path1 = sys.argv[1]
    score_path2 = sys.argv[2]
    # print
    l1 = get_score_per_query(score_path1)
    l2 = get_score_per_query(score_path2)

    assert len(l1) == len(l2)

    d, p_value = stats.ttest_rel(l1, l2)
    print("baseline:", average(l1))
    print("treatment:", average(l2))
    print(d, p_value)


if __name__ == "__main__":
    main()