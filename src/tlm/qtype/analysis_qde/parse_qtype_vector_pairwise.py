from typing import List, Tuple

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