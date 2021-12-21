from typing import Iterator
from typing import Iterator

import numpy as np

from misc_lib import group_by
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance


def print_variance(qtype_entries: Iterator[QTypeInstance], query_info_dict: Dict[str, QueryInfo]):
    def get_func_str(e: QTypeInstance) -> str:
        func_str = " ".join(query_info_dict[e.qid].functional_tokens)
        return func_str

    grouped = group_by(qtype_entries, get_func_str)
    for func_str, items in grouped.items():

        all_qe_vector = np.stack([e.qtype_weights_qe for e in items], axis=0)
        for e in items:
            query0 = " ".join(query_info_dict[e.qid].out_s_list)
            sample_de = e.qtype_weights_de
            scores = np.matmul(all_qe_vector, sample_de)
            score_avg = np.average(scores)
            if score_avg > 0:
                score_std = np.std(scores)
                n_score = len(scores)
                error = np.abs(scores - score_avg)
                threshold = 3
                n_large_error = np.count_nonzero(np.less(threshold, error))
                print(f"avg={score_avg:.2f} std={score_std:.2f}, {n_large_error} errors among {n_score} : {query0}")
                break
