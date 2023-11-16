import heapq
import random
import sys
from typing import List

from omegaconf import OmegaConf

from list_lib import apply_batch
from misc_lib import path_join, TimeEstimatorOpt
from taskman_client.job_group_proxy import SubJobContext
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import get_term_pair_predictor


def read_lines(path):
    lines = open(path, "r").readlines()
    return [l.strip() for l in lines]


def predict_with_fixed_context_model_and_save(
        predict_term_pairs_fn, q_term: str, d_term_list: List[str], log_path,
        outer_batch_size):
    n_item = len(d_term_list)
    n_batch = n_item // outer_batch_size
    n_keep = 5000

    min_heap = []
    ticker = TimeEstimatorOpt(n_batch)
    for batch_terms in apply_batch(d_term_list, outer_batch_size):
        pairs = [(q_term, d_term) for d_term in batch_terms]
        scores = predict_term_pairs_fn(pairs)

        for d_term, score in zip(batch_terms, scores):
            # Push item with its negative score (to use min heap as max heap)
            heapq.heappush(min_heap, (score, d_term))
            # If the heap size exceeds k, remove the smallest element (which is the largest negative score)
            if len(min_heap) > n_keep:
                heapq.heappop(min_heap)

        ticker.tick()

    save_items = []
    for _ in range(len(min_heap)):
        neg_score, d_term = heapq.heappop(min_heap)
        score = neg_score
        save_items.append((d_term, score))
    save_items = save_items[::-1]
    save_tsv(save_items, log_path)


def main():
    conf_path = sys.argv[1]
    job_no = int(sys.argv[2])
    conf = OmegaConf.load(conf_path)
    q_terms = read_lines(conf.q_term_path)
    d_terms = read_lines(conf.d_term_path)
    job_size = int(conf.job_size)
    job_name = conf.run_name
    model_path = conf.model_path
    save_dir = conf.save_dir


    num_items = len(q_terms)
    max_job = num_items

    num_job_per_slurm_job = job_size
    num_slurm_job = max_job // num_job_per_slurm_job + 1
    st = job_no * job_size
    ed = st + job_size
    for q_term_i in range(st , ed):
        log_path = path_join(save_dir, f"{job_no}.txt")
        predict_term_pairs_fn = get_term_pair_predictor(model_path)
        with SubJobContext(job_name, job_no, max_job):
            predict_with_fixed_context_model_and_save(
                predict_term_pairs_fn, q_terms[q_term_i], d_terms,
                log_path, outer_batch_size=100)


if __name__ == "__main__":
    main()


# 1K/per min