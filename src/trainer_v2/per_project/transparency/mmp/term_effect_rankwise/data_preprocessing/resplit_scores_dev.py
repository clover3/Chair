import csv
import os

from misc_lib import SuccessCounter
from taskman_client.wrapper3 import JobContext
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import read_shallow_scores_train, \
    get_shallow_score_save_path_by_qid, get_deep_score_save_path_by_qid, load_qtfs_train, load_qtfs, \
    get_shallow_score_save_path, read_qid_pid_score_triplet_grouped
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition_for_train, \
    get_mmp_split_w_deep_scores_train, get_mmp_split_w_deep_scores, get_valid_mmp_partition_for_dev
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import load_deep_scores


def save_tsv(output, save_path):
    tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
    tsv_writer.writerows(output)


def do_for_shallow():
    split = "dev"
    for job_id in get_valid_mmp_partition_for_dev():
        any_fail = False
        for qid, _ in load_qtfs(split, job_id):
            check_path = get_shallow_score_save_path_by_qid(qid)
            if not os.path.exists(check_path):
                any_fail = True
                break

        if not any_fail:
            print(job_id, "Skip")
            continue
        else:
            print(job_id, "Work")

        save_path = get_shallow_score_save_path(split, job_id)
        output: List[Tuple[str, List[Tuple[str, float]]]] = \
            read_qid_pid_score_triplet_grouped(save_path)
        for qid, entries in output:
            save_path = get_shallow_score_save_path_by_qid(qid)
            save_tsv(entries, save_path)


def do_for_deep():
    print("do for deep")
    split = "dev"
    for job_no in get_mmp_split_w_deep_scores(split):
        any_fail = False
        for qid, _ in load_qtfs(split, job_no):
            check_path = get_deep_score_save_path_by_qid(qid)
            if not os.path.exists(check_path):
                any_fail = True
                break

        if not any_fail:
            print(job_no, "Skip")
            continue

        print(job_no, "Work")
        deep_score_grouped: List[List] = load_deep_scores(split, job_no)
        for group in deep_score_grouped:
            qid, _, _ = group[0]
            save_path = get_deep_score_save_path_by_qid(qid)
            save_tsv(group, save_path)


def main():
    with JobContext("split deep/shallow scores"):
        do_for_shallow()


if __name__ == "__main__":
    main()