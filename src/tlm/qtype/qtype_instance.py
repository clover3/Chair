from typing import NamedTuple

import numpy as np


class QTypeInstance(NamedTuple):
    qid: str
    doc_id: str
    passage_idx: str
    de_input_ids: np.array
    qtype_weights_qe: np.array
    qtype_weights_de: np.array
    label: int
    logits: float
    bias: float
    q_bias: float
    d_bias: float