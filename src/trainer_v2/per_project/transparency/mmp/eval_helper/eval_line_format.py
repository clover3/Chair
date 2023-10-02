from typing import Iterable

from list_lib import apply_batch
from misc_lib import TELI, ceil_divide, TimeEstimator


def predict_qd_itr_save_score_lines(score_fn, itr, scores_path, data_size):
    f = open(scores_path, "w")
    if data_size:
        itr = TELI(itr, data_size)
    for q, d in itr:
        score = score_fn(q, d)
        f.write("{}\n".format(score))


def score_and_save_score_lines(itr, score_fn, scores_path, max_batch_size, data_size):
    f = open(scores_path, "w")
    if data_size:
        n_batch = ceil_divide(data_size, max_batch_size)
        ticker = TimeEstimator(n_batch)
    for batch in apply_batch(itr, max_batch_size):
        scores: Iterable[float] = score_fn(batch)
        for x, s in zip(batch, scores):
            f.write("{}\n".format(s))

        if data_size:
            ticker.tick()