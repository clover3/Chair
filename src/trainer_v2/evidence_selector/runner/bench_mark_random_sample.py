import random
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
from cache import load_from_pickle
from data_generator2.segmented_enc.es.evidence_candidate_gen import pool_delete_indices
from misc_lib import ceil_divide

from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed
from trainer_v2.evidence_selector.defs import RLStateTensor
import numpy as np
c_log.info("begin")
state_list = load_from_pickle("state_list")

n_random = 10

def get_seg2_start(segment_ids):
    for i, t in enumerate(segment_ids):
        if segment_ids[i] == 1:
            return i
    return len(segment_ids)


def delete_seg1(input_ids, segment_ids) -> List[int]:
    # seg2_st, _ = get_st_ed(segment_ids)
    seg2_st = 10
    n_tokens = seg2_st - 2
    g = 0.5
    g_inv = int(1 / g)
    max_del = ceil_divide(n_tokens, g_inv)
    num_del = random.randint(1, max_del)
    indices: List[int] = pool_delete_indices(num_del, n_tokens, g)
    items = [1] * len(input_ids)
    for i in indices:
        items[i+1] = 0

    return items


def get_random_sample(state_list: List[RLStateTensor]) -> List[List[int]]:
    def state_to_sample(state: RLStateTensor):
        return delete_seg1(state.input_ids, state.segment_ids)

    # Return: [B, K, L]
    return list(map(state_to_sample, state_list))


c_log.info("before")
for _ in range(n_random):
    result = get_random_sample(state_list)
# result = []
# for state in state_list:
#     seg2_st = get_seg2_start(state.segment_ids)
#     a_list = []
#     for _ in range(n_random):
#         a = delete_seg1(state.input_ids, seg2_st)
#         a_list.append(a)
#     result.append(a_list)

c_log.info("after")
