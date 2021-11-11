from collections import Counter
from typing import List, Tuple

import numpy as np

from cache import save_to_pickle
from cpath import at_output_dir
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list, convert_ids_to_tokens
from tlm.qtype.qtype_analysis import QTypeInstance2


def parse_q_weight_output(raw_prediction_path):
    viewer = EstimatorPredictionViewer(raw_prediction_path)
    tokenizer = viewer.tokenizer
    voca_list = get_voca_list(tokenizer)
    out_entries: List[Tuple[QTypeInstance2, QTypeInstance2]] = []
    for e in viewer:
        # ticker.tick()
        def get_vector_wrap(key, post_fix):
            return e.get_vector(key + post_fix)

        def parse_one(post_fix):
            qe_input_ids = get_vector_wrap("qe_input_ids", post_fix)
            de_input_ids = get_vector_wrap("de_input_ids", post_fix)
            qtype_vector_qe = get_vector_wrap("qtype_vector_qe", post_fix)
            qtype_vector_de = get_vector_wrap("qtype_vector_de", post_fix)

            def split_conv_input_ids(input_ids):
                seg1, seg2 = split_p_h_with_input_ids(input_ids, input_ids)
                seg1_tokens = convert_ids_to_tokens(voca_list, seg1)
                seg2_tokens = convert_ids_to_tokens(voca_list, seg2)
                return seg1_tokens, seg2_tokens

            entity, query = split_conv_input_ids(qe_input_ids)
            entity, doc = split_conv_input_ids(de_input_ids)
            inst = QTypeInstance2(query, entity, doc, qtype_vector_qe, qtype_vector_de,
                                   int(post_fix == "1"))
            return inst

        e = parse_one("1"), parse_one("2")
        out_entries.append(e)
    return out_entries


def vector_dim_analysis(all_insts: List[QTypeInstance2]):
    n_cluster = 1000
    clusters = [list() for _ in range(n_cluster)]
    for j, inst in enumerate(all_insts):
        qtype_weights = inst.qtype_weights_q
        ranked_dims = np.argsort(qtype_weights)[::-1]
        for d in ranked_dims[:10]:
            if qtype_weights[d] > 0.5:
                c: List = clusters[d]
                c.append(j)

            # s = "{0}({1:.2f})".format(d, qtype_weights[d])
            # s_l.append(s)
        # print(inst.label, inst.summary())
        # print(" ".join(s_l))

    for c_idx, c in enumerate(clusters):
        if not len(c):
            continue

        print("Cluster {}".format(c_idx))
        print("# Distinct queries :", len(c))
        print("<<<")
        s_counter = Counter()
        for j in c:
            s = all_insts[j].summary()
            s_counter[s] += 1
        for s, cnt in s_counter.most_common():
            print(cnt, s)


def main():
    run_name_list = ["mmd_4U"]
    for run_name in run_name_list:
        raw_prediction_path = at_output_dir("qtype", run_name)
        out_entries = parse_q_weight_output(raw_prediction_path)
        save_to_pickle(out_entries, run_name + "_qtype_parsed")
        out_entries_flat = []
        for e1, e2 in out_entries:
            out_entries_flat.append(e1)
            out_entries_flat.append(e2)
        vector_dim_analysis(out_entries_flat)


if __name__ == "__main__":
    main()