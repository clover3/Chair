import csv
import gzip
import sys
from cpath import output_path, data_path, get_canonical_model_path
from table_lib import tsv_iter
from misc_lib import path_join, TimeEstimator
from misc_lib import select_third_fourth
from taskman_client.task_proxy import get_task_proxy
from trainer_v2.chair_logging import c_log
from list_lib import apply_batch
from trainer_v2.per_project.transparency.mmp.rerank import get_scorer
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    job_no = sys.argv[1]
    proxy = get_task_proxy()
    split = "dev"
    run_name = f"score_mmp_{split}_{job_no}"
    proxy.task_start(run_name)
    quad_tsv_path = path_join(data_path, "msmarco", "passage", f"{split}_group_sorted_10K", f"{job_no}")
    model_path = get_canonical_model_path("mmp1")
    scores_path = path_join(output_path, "msmarco", "passage", f"mmp_{split}_split_all_scores", f"{job_no}.scores")

    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    flush_block_size = 1024
    batch_size = 256
    data_size = 10 * 10000

    f = open(scores_path, "w")
    strategy = get_strategy()
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn = get_scorer(model_path, batch_size)
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

