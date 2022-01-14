import os
from typing import Dict

import numpy as np

from bert_api.client_lib import BERTClient
from cache import load_pickle_from
from cpath import data_path, pjoin
from cpath import output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import convert_ids_to_tokens, get_tokenizer, EncoderUnitPlain, pretty_tokens
from port_info import FDE_PORT
from tlm.data_gen.doc_encode_common import split_by_window
from tlm.qtype.analysis_fde.analysis_a import embeddings_to_list
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_emb_qtype_2X_v_train_200000, load_q_bias
from tlm.qtype.analysis_qde.contribution_module import print_base_info
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.contribution_common import enum_window_drop, contribution_by_change
from tlm.qtype.enum_util import enum_samples, enum_interesting_entries


class FDEClientWrap:
    def __init__(self):
        max_seq_length = 512
        self.client = BERTClient("http://localhost", FDE_PORT, max_seq_length)

        voca_path = pjoin(data_path, "bert_voca.txt")
        self.q_encoder = EncoderUnitPlain(128, voca_path)
        self.d_encoder = EncoderUnitPlain(max_seq_length, voca_path)

    def request(self, seg1, seg2):
        def flat(d):
            return d["input_ids"], d["input_mask"], d["segment_ids"]

        qe_input_ids, qe_input_mask, qe_segment_ids = self.q_encoder.encode_pair("", "")
        de_input_ids, de_input_mask, de_segment_ids = flat(self.d_encoder.encode_inner(seg1, seg2))
        one_inst = qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids
        payload_list = [one_inst]
        ret = self.client.send_payload(payload_list)[0]
        return ret


def run_contribution_analysis(qtype_entries,
                              query_info_dict: Dict[str, QueryInfo],
                              q_embedding_d: Dict[str, np.array],
                              q_bias_d: Dict[str, np.array],
                              ):
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    unknown_token = tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
    window_size = 20
    fde_client = FDEClientWrap()
    empty_func_span = "[MASK]"
    func_span_list, qtype_embedding_np = embeddings_to_list(q_embedding_d)

    def compute_score(func_span, q_bias_d, seg1_np, seg2):
        ret = fde_client.request(seg1_np, seg2)
        doc_vector = ret['qtype_vector2']
        d_bias = ret['d_bias']
        empty_q_vector = q_embedding_d[empty_func_span]
        target_q_vector = q_embedding_d[func_span]
        score = np.dot(target_q_vector, doc_vector) + d_bias + q_bias_d[func_span]
        score_type_less = np.dot(empty_q_vector, doc_vector) + d_bias + q_bias_d[empty_func_span]
        return score, score_type_less

    for e in enum_interesting_entries(qtype_entries, query_info_dict):
        info = query_info_dict[e.qid]
        input_ids = e.de_input_ids
        seg1_np, seg2_np = split_p_h_with_input_ids(input_ids, input_ids)
        seg1 = seg1_np.tolist()
        seg2 = seg2_np.tolist()

        func_span = info.get_func_span_rep()
        if func_span not in q_embedding_d:
            print(func_span, "NOT FOUND")
            continue
        base_score, base_score_typeless = compute_score(func_span, q_bias_d, seg1, seg2)
        single_sent_list = list(split_by_window(seg2, window_size))
        dropped_seg2s = list(enum_window_drop(seg2, unknown_token, window_size))

        sent_drop_result_pairs = [compute_score(func_span, q_bias_d, seg1, window) for window in dropped_seg2s]
        single_sent_result_pairs = [compute_score(func_span, q_bias_d, seg1, window) for window in single_sent_list]

        sent_drop_result, sent_drop_result_typeless = zip(*sent_drop_result_pairs)
        single_sent_result, single_sent_result_typeless = zip(*single_sent_result_pairs)

        contrib_single = contribution_by_change(base_score,
                                                single_sent_result)
        contrib_type_less_single = contribution_by_change(base_score_typeless,
                                                          single_sent_result_typeless, )

        contrib_drop = contribution_by_change(base_score,
                                              sent_drop_result)
        contrib_drop_type_less = contribution_by_change(base_score_typeless,
                                                        sent_drop_result_typeless, )
        print_base_info(e, query_info_dict)
        print("Base score: {0:.2f}".format(base_score))
        head0 = ['by single', '', '', 'by drop', '', '']
        head1 = ['full', 'no_func', 'diff', 'full', 'no_func', 'diff']
        print("\t".join(head0))
        print("\t".join(head1))
        for window_idx, window in enumerate(split_by_window(seg2, window_size)):
            seg2_tokens = convert_ids_to_tokens(voca_list, window)
            passage: str = pretty_tokens(seg2_tokens, True)
            numbers = [
                contrib_single[window_idx],
                contrib_type_less_single[window_idx],
                contrib_single[window_idx] - contrib_type_less_single[window_idx],
                contrib_drop[window_idx],
                contrib_drop_type_less[window_idx],
                contrib_drop[window_idx] - contrib_drop_type_less[window_idx],
                ]
            s = "\t".join(["{0:.2f}".format(v) for v in numbers]) + "\t" + passage
            print(s)
        print("")


def main():
    # run_name = "qtype_2X_v_train_200000"
    run_name = "qtype_2Y_v_train_120000"
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype_2X_v_train_200000()
    save_dir = os.path.join(output_path, "qtype", run_name + '_sample')
    _, query_info_dict = load_pickle_from(os.path.join(save_dir, "0"))
    qtype_entries = enum_samples(save_dir)
    q_bias_d: Dict[str, np.array] = load_q_bias(run_name)
    run_contribution_analysis(qtype_entries, query_info_dict, q_embedding_d, q_bias_d)


if __name__ == "__main__":
    main()
