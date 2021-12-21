from collections import Counter
from typing import List

import numpy as np

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer, convert_ids_to_tokens, pretty_tokens
from tlm.qtype.analysis_fixed_qtype.parse_dyn_qtype_vector import show_vector_distribution
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list
from tlm.qtype.qtype_instance import QTypeInstance


def analysis_b(qtype_entries: List[QTypeInstance],
                       query_info_dict,
                       known_qtype_ids,
                       scale_factor_list):
    scale_factor = np.array(scale_factor_list)
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    print("Building qtype desc")
    print("Done")
    threshold = 0.5
    pos_known, neg_known = known_qtype_ids

    print("Done")
    def print_by_dimension(v):
        rank = np.argsort(v)[::-1]
        for i in rank:
            value = v[i]
            if value < threshold:
                break
            else:
                if i in pos_known:
                    print("{0}: {1:.2f}".format(i, value))

    def print_by_dimension_neg(v):
        rank = np.argsort(v)
        for i in rank:
            value = v[i]
            if value > -threshold:
                break
            else:
                if i in neg_known:
                    print("{0}: {1:.2f}".format(i, value))



    print_per_qid = Counter()
    n_print = 0
    for e_idx, e in enumerate(qtype_entries):
        display = False
        if e_idx % 10 and print_per_qid[e.qid] < 4:
            pass
        #  : Show qid/query text
        #  : Show non-zero score of document
        if e.logits > 3 or e.d_bias > 3:
            why_display = "Display by high logits"
            display = True
        if e.label:
            why_display = "Display by true label"
            display = True
        if not display:
            continue
        print_per_qid[e.qid] += 1

        q_rep = " ".join(query_info_dict[e.qid].out_s_list)

        scaled_query_qtype = e.qtype_weights_qe / scale_factor
        scaled_document_qtype = e.qtype_weights_de * scale_factor
        display = False
        for key in pos_known:
            if scaled_document_qtype[key] > threshold:
                display = True

        for key in neg_known:
            if scaled_document_qtype[key] < -threshold:
                display = True
        if not display:
            continue
        n_print += 1
        if n_print % 5 == 0:
            dummy = input("Enter something ")

        print("---------------------------------")
        print(e.qid, q_rep)
        print("{}:{} - {}".format(e.doc_id, e.passage_idx, "Relevant" if e.label else "Non-relevant"))
        print("Score bias q_bias d_bias")
        print(" ".join(map("{0:.2f}".format, [e.logits, e.bias, e.q_bias, e.d_bias])))
        # show_vector_distribution(scaled_query_qtype)
        # print_by_dimension(scaled_query_qtype)
        print("Doc QType Vector")
        show_vector_distribution(scaled_document_qtype)
        print_by_dimension(scaled_document_qtype)
        print_by_dimension_neg(scaled_document_qtype)

        print(e.qtype_weights_de)
        seg1, seg2 = split_p_h_with_input_ids(e.de_input_ids, e.de_input_ids)
        seg2_tokens = convert_ids_to_tokens(voca_list, seg2)
        passage: str = pretty_tokens(seg2_tokens, True)
        print(passage)

