import pickle
import sys
from collections import Counter

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, enum_grouped2
from dataset_specific.msmarco.passage.path_helper import get_mmp_train_grouped_sorted_path
from list_lib import assert_length_equal
from misc_lib import select_first_second
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.data_preprocessing.serializer import \
    save_shallow_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_tfs_and_computed_base_scores, \
    get_tfs_save_path, get_shallow_score_save_path
from typing import List, Iterable, Tuple


def save_shallow_scores_train(
        job_id,
        qid_scores: List[Tuple[str, List[Tuple[str, float]]]]):
    split = "train"
    save_path = get_shallow_score_save_path(split, job_id)
    save_shallow_scores(save_path, qid_scores)


def save_qid_tfs(qid, qid_tfs):
    save_path = get_tfs_save_path(qid)
    pickle.dump(qid_tfs, open(save_path, "wb"))


def resplit_tfs_inner(job_no):
    c_log.info("load_tfs_and_computed_base_scores")
    tfs_and_shallow: List[List[Tuple[str, Counter, float]]] = load_tfs_and_computed_base_scores(job_no)
    quad_tsv_path = get_mmp_train_grouped_sorted_path(job_no)
    c_log.info("load quad tsv")
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(quad_tsv_path)))
    c_log.info("load group enum")
    qid_pid_grouped: Iterable[List[Tuple]] = list(enum_grouped2(qid_pid))

    c_log.info("enum writing...")
    qid_scores: List[Tuple[str, List[Tuple[str, float]]]] = []
    for tf_shallow_group, qid_pid_group in zip(tfs_and_shallow, qid_pid_grouped):
        assert_length_equal(tf_shallow_group, qid_pid_group)
        qid, _ = qid_pid_group[0]

        scores: List[Tuple[str, float]] = []
        tfs_list: List[Tuple[str, Counter]] = []
        for (pid1, tfs, score), (qid, pid2) in zip(tf_shallow_group, qid_pid_group):
            assert pid1 == pid2
            scores.append((pid1, score))
            tfs_list.append((pid1, tfs))

        qid_scores.append((qid, scores))

        qid_tfs = qid, tfs_list
        save_qid_tfs(qid, qid_tfs)

    c_log.info("Done enum saving shallow scores...")
    save_shallow_scores_train(job_no, qid_scores)
    c_log.info("Done")


def main():
    job_no = int(sys.argv[1])
    run_name = "resplit_tfs_{}".format(job_no)

    with JobContext(run_name):
        resplit_tfs_inner(job_no)


if __name__ == "__main__":
    main()
