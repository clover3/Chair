import os
from typing import List, Iterable

from cache import load_pickle_from
from cpath import output_path
from tlm.qtype.analysis_fde.q_emb_tool import get_vector_diff
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.qtype_instance import QTypeInstance


def run_print():
    # MMD_train_qe_de_distill_base_prob
    run_name = "qtype_2X_v_train_200000"
    pjoin = os.path.join
    sample_save_dir = pjoin(output_path, "qtype", run_name + '_sample')
    qtype_entries0, query_info_dict0 = load_pickle_from(pjoin(sample_save_dir, "0"))
    qtype_entries1, query_info_dict1 = load_pickle_from(pjoin(sample_save_dir, "1"))
    target_func_span = "who is"
    qtype_entries0: List[QTypeInstance] = qtype_entries0
    qtype_entries1: List[QTypeInstance] = qtype_entries1

    def is_target(e):
        info: QueryInfo = query_info_dict0[e.qid]
        func_span = " ".join(info.functional_tokens)
        return func_span == target_func_span

    targets_a: Iterable[QTypeInstance] = filter(is_target, qtype_entries0)
    targets_b: Iterable[QTypeInstance] = filter(is_target, qtype_entries1)
    prev_v = None
    for e in targets_b:
        info: QueryInfo = query_info_dict0[e.qid]
        print(e.qid, info.query)
        if prev_v is not None:
            error = get_vector_diff(prev_v, e.qtype_weights_qe)
            if error > 1e-2:
                print(error)


def main():
    run_print()


if __name__ == "__main__":
    main()



