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
    counter = Counter()

    file_list = get_dir_files(path_helper.per_pair_candidates.fidelity_save_dir)
    file_name_list = [os.path.basename(t) for t in file_list]
    do_detail = False
    ticker = TimeEstimator(len(todo_list))
    not_done_list = []
    for idx, (q_term, d_term) in enumerate(todo_list):
        ticker.tick()
        save_name = str(idx)

        if save_name in file_name_list:
            counter["pair done"] += 1
        else:
            if not do_detail:
                continue
            last_item = None
            for job_no in get_valid_mmp_partition_for_train():
                file_name = f"{idx}_{job_no}.jsonl.gz"
                te_path = path_join(
                    path_helper.per_pair_candidates.term_effect_save_dir, file_name)
                if os.path.exists(te_path):
                    last_item = job_no
                else:
                    break

            if last_item is None:
                counter["not started"] += 1
            else:
                not_done_list.append((idx, last_item))
                counter["started not done"] += 1

    tab_print_dict(counter)


if __name__ == "__main__":
    main()