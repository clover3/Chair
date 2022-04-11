import abc
from typing import List, Dict, Tuple
import numpy as np
from scipy.special import softmax


INPUT_IDS = List[int]
INPUT_MASK = List[int]
SEGMENT_IDS = List[int]
ATTENTION_MASK = Dict[Tuple[int, int], int]
SinglePayload = Tuple[INPUT_IDS, INPUT_MASK, SEGMENT_IDS, ATTENTION_MASK]


def serialize_tuple_d(payload: List[SinglePayload]):
    def conv_key(k):
        idx1, idx2 = k
        return "{},{}".format(idx1, idx2)

    def conv_inst(e):
        x0, x1, x2, x3 = e
        x3 = {conv_key(k): v for k, v in x3.items()}
        return x0, x1, x2, x3

    return [conv_inst(e) for e in payload]


def deserialize_tuple_d(payload):
    def rev_conv_key(k):
        idx1, idx2 = k.split(",")
        return int(idx1), int(idx2)

    def conv_inst(e):
        x0, x1, x2, x3 = e
        x3 = {rev_conv_key(k): v for k, v in x3.items()}
        return x0, x1, x2, x3

    return [conv_inst(e) for e in payload]


class BERTMaskIF(abc.ABC):
    @abc.abstractmethod
    def predict(self, items: List[SinglePayload]):
        pass


def logits_to_score(logits):
    return softmax(logits)[1]



def later_score_prob(base, after):
    return logits_to_score(after)


def dist_l2(base: List[float], after: List[float]):
    return np.linalg.norm(np.array(base) - np.array(after))