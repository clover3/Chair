import os
from typing import List, Iterable, Callable, Dict, Tuple, Set

from list_lib import lmap
from misc_lib import average
from tlm.data_gen.robust_gen.stat_tets import get_score_per_file

items = ["robust_3B", "robust_3A",
"robust_3I2",
"robust_3AC",
"robust_3U",
"robust_3S",
"robust_3Q",
"robust_3T",]


def main():
    common_dir = os.path.join('work', "ndcg")
    def load_scores(run_name):
        scores_list = []
        for i in range(5):
            file_name = "{}_combine_{}.txt".format(run_name, i)
            scores: List[float] = get_score_per_file(os.path.join(common_dir, file_name))
            scores_list.append(scores)
        return scores_list

    for run in items:
        print(run)
        score_per_repeat = lmap(average, load_scores(run))
        print("{0:.3f}".format(average(score_per_repeat)))


if __name__ == "__main__":
    main()