import os.path
from collections import Counter

from misc_lib import path_join, TimeEstimator, get_dir_files
from tab_print import tab_print_dict
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand4_path_helper
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition_for_train


def main():
    path_helper = get_cand4_path_helper()
    path_helper.load_qterm_candidates()
    todo_list: List[Tuple] = list(tsv_iter(path_helper.per_pair_candidates.candidate_pair_path))

    file_list = get_dir_files(path_helper.per_pair_candidates.fidelity_save_dir)
    ticker = TimeEstimator(len(todo_list))
    for f_path in file_list:
        name = os.path.basename(f_path)
        idx = int(name)
        q_term, d_term = todo_list[idx]
        value = float(open(f_path, "r").read())
        if value > 0:
            print(q_term, d_term, value)


if __name__ == "__main__":
    main()
