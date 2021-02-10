from typing import NamedTuple, List

import numpy as np
import scipy.special

from cache import load_pickle_from


class OutputViewer:
    def __init__(self, save_path, n_factor, batch_size):
        self.raw_content = load_pickle_from(save_path)
        self.n_factor = n_factor
        self.batch_size = batch_size

    def __iter__(self):
        for batch in self.raw_content:
            data_id = batch['data_id']
            doc = batch['doc']
            raw_logits = batch['logits']
            try:
                logits = np.reshape(raw_logits, [-1, self.n_factor, 2])
                for j in range(self.batch_size):
                    d = {
                        'data_id': str(data_id[j][0]),
                        'doc': doc[j],
                        'logits': logits[j],
                    }
                    yield d
            except ValueError:
                pass


def get_probs(logits):
    probs = scipy.special.softmax(logits, axis=-1)
    return probs[:, 1]


def get_piece_scores(n_factor, probs, segment_len, step_size):
    seg_scores = []
    for i in range(n_factor):
        st = i * step_size
        ed = st + segment_len
        e = st, ed, probs[i]
        seg_scores.append(e)
    return seg_scores


class ScoredPiece(NamedTuple):
    st: int
    ed: int
    score: float
    tokens: List[str]


class ScoredPieceFromPair(NamedTuple):
    st: int
    ed: int
    score: float
    raw_score1: float
    raw_score2: float
    tokens: List[str]