import sys
from typing import Iterable, Tuple

from cpath import output_path, data_path
from list_lib import apply_batch
from misc_lib import path_join, TimeEstimator
from misc_lib import select_third_fourth
from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from taskman_client.task_proxy import get_task_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.train_all_inf.score_from_gz import tsv_iter_from_gz


def main():
    job_no = sys.argv[1]
    proxy = get_task_proxy()
    run_name = f"score_mmp_{job_no}"
    proxy.task_start(run_name)

    partition_no = job_no % 100
    quad_tsv_path = path_join(data_path, "msmarco", "passage", "group_sorted_10K_gz", f"{job_no}.gz")
    scores_path = path_join(
        output_path, "msmarco", "passage",
        "mmp_train_split_all_scores_tf",
        f"{job_no}_{partition_no}.scores")
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter_from_gz(quad_tsv_path))
    flush_block_size = 64
    data_size = 1000 * 10000
    f = open(scores_path, "w")

    c_log.info("Building scorer")
    score_fn = get_ce_msmarco_mini_lm_score_fn()
    n_batch = int(data_size / flush_block_size)
    ticker = TimeEstimator(n_batch)
    for batch in apply_batch(tuple_itr, flush_block_size):
        scores = score_fn(batch)
        for x, s in zip(batch, scores):
            f.write("{}\n".format(s))
        ticker.tick()

    proxy.task_complete(run_name)


if __name__ == "__main__":
    main()
