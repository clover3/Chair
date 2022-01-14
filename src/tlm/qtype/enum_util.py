import os
from typing import List

from cache import load_pickle_from
from tlm.qtype.qtype_instance import QTypeInstance


def enum_samples(save_dir):
    num_jobs = 37
    for i in range(num_jobs):
        try:
            insts, _ = load_pickle_from(os.path.join(save_dir, str(i)))
            yield from insts
        except FileNotFoundError as e:
            print(e)


def enum_interesting_entries(qtype_entries: List[QTypeInstance], query_info_dict):
    # qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    for e_idx, e in enumerate(qtype_entries):
        f_high_logit = e.logits > 0 or e.d_bias > 2

        display = False
        if f_high_logit:
            display = True

        info = query_info_dict[e.qid]
        content_tokens = info.content_span.split()
        f_short_content = len(content_tokens) < 3
        if not f_short_content:
            display = False

        if display:
            yield e