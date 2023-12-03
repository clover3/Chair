import heapq
import os.path
from typing import List

from list_lib import apply_batch
from misc_lib import TimeEstimatorOpt
from trainer_v2.per_project.transparency.misc_common import save_tsv


def predict_save_top_k(
        predict_term_pairs_fn,
        q_term: str,
        d_term_list: List[str],
        log_path,
        outer_batch_size, n_keep = 5000):

    if os.path.exists(log_path):
        print(f"{log_path} exists ")
        return
    n_item = len(d_term_list)
    n_batch = n_item // outer_batch_size


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