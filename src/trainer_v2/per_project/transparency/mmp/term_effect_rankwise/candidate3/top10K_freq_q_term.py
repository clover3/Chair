from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import TEL
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_qtfs
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_mmp_galign_path_helper, \
    MMPGAlignPathHelper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition_for_train


def main():
    split = "train"
    config: MMPGAlignPathHelper = get_mmp_galign_path_helper()

    qt_df = Counter()
    for i in get_valid_mmp_partition_for_train():
        qtfs: List[Tuple[str, Counter]] = load_qtfs(split, i)
        for qid, query_tfs in qtfs:
            for q_term in query_tfs:
                qt_df[q_term] += 1

    f = open(config.path_config.frequent_q_terms, "w")
    selected = []
    for t, cnt in qt_df.most_common(10000):
        selected.append(t)
        f.write(t + "\n")

    print(f"Selected {len(selected)} terms, while the last is ({t}, {cnt})")



if __name__ == "__main__":
    main()