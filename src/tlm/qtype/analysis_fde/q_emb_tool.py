from collections import defaultdict
from typing import Iterator, Dict, Tuple

import numpy as np

from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import merge_vectors_by_frequency
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance


def get_vector_diff(v1, v2):
    mag = (sum(np.abs(v1)) + sum(np.abs(v2))) / 2
    error = np.sum(np.abs(v1 - v2))
    rel_error = error / mag
    return error, rel_error


def build_qtype_embedding(qtype_entries: Iterator[QTypeInstance], query_info_dict: Dict[str, QueryInfo]) \
        -> Dict[Tuple[str, str], np.array]:

    func_str_to_vector = defaultdict(list)
    for e in qtype_entries:
        head, tail = query_info_dict[e.qid].get_head_tail
        rep = " ".join(head), " ".join(tail)
        func_str_to_vector[rep].append(e.qtype_weights_qe)

    out_d = apply_merge(func_str_to_vector)
    return out_d


def apply_merge(func_str_to_vector) -> Dict[Tuple[str], np.array]:
    out_d = {k: merge_vectors_by_frequency(v) for k, v in func_str_to_vector.items()}
    return out_d