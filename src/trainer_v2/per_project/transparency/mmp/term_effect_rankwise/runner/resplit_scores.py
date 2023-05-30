import csv

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import read_shallow_scores, \
    get_shallow_score_save_path_by_qid, get_deep_score_save_path_by_qid
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.check_mmp_train_split_all_scores import \
    get_valid_mmp_split, get_mmp_split_w_deep_scores
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import load_deep_scores


def save_tsv(output, save_path):
    tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
    tsv_writer.writerows(output)


def do_for_shallow():
    for job_id in get_valid_mmp_split():
        print(job_id)
        output: List[Tuple[str, List[Tuple[str, float]]]] = read_shallow_scores(job_id)
        for qid, entries in output:
            save_path = get_shallow_score_save_path_by_qid(qid)
            save_tsv(entries, save_path)


def do_for_deep():
    print("do for deep")
    for job_no in get_mmp_split_w_deep_scores():
        print(job_no)
        deep_score_grouped: List[List] = load_deep_scores(job_no)
        for group in deep_score_grouped:
            qid, _, _ = group[0]
            save_path = get_deep_score_save_path_by_qid(qid)
            save_tsv(group, save_path)


def main():
    do_for_deep()


if __name__ == "__main__":
    main()