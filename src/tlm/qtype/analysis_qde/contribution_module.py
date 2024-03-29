from typing import List

import numpy as np

from bert_api.client_lib import BERTClient
from cache import load_from_pickle
from cpath import data_path, pjoin
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import convert_ids_to_tokens, get_tokenizer, EncoderUnitPlain, pretty_tokens
from port_info import QDE_PORT, MMD_Z_PORT
from tlm.data_gen.doc_encode_common import split_by_window
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list
from tlm.qtype.contribution_common import contribution_by_change, enum_window_drop
from tlm.qtype.enum_util import enum_interesting_entries
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import softmax_rev_sigmoid


class QDEClientWrap:
    def __init__(self):
        max_seq_length = 512
        self.client = BERTClient("http://localhost", QDE_PORT, max_seq_length)

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


class MMD_Z:
    def __init__(self):
        max_seq_length = 512
        self.client = BERTClient("http://localhost", MMD_Z_PORT, max_seq_length)

        voca_path = pjoin(data_path, "bert_voca.txt")
        self.encoder = EncoderUnitPlain(max_seq_length, voca_path)

    def request(self, seg1: List[int], seg2: List[int]):
        def flat(d):
            return d["input_ids"], d["input_mask"], d["segment_ids"]

        one_inst = flat(self.encoder.encode_inner(seg1, seg2))
        payload_list = [one_inst]
        ret = self.client.send_payload(payload_list)
        return ret


class MMD_Z_RevSigmoid:
    def __init__(self):
        self.inner = MMD_Z()

    def request(self, seg1: List[int], seg2: List[int]):
        ret = self.inner.request(seg1, seg2)
        return softmax_rev_sigmoid(ret)[0]


def run_contribution_analysis(qtype_entries, query_info_dict):
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    unknown_token = tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
    window_size = 20
    qde_client = QDEClientWrap()
    mmd_client = MMD_Z_RevSigmoid()

    def qde_forward_run(seg1, seg2) -> float:
        ret = qde_client.request(seg1, seg2)
        empty_qtype_vector1 = ret['qtype_vector1']
        qtype_vector2 = ret['qtype_vector2']
        score = np.dot(e.qtype_weights_qe, qtype_vector2)
        return float(score)

    def qde_forward_run_typeless(seg1, seg2) -> float:
        ret = qde_client.request(seg1, seg2)
        empty_qtype_vector1 = ret['qtype_vector1']
        qtype_vector2 = ret['qtype_vector2']
        score = np.dot(empty_qtype_vector1, qtype_vector2)
        return float(score)

    def mmd_forward_run(seg1, seg2) -> float:
        ret = mmd_client.request(seg1, seg2)
        return ret

    forward_run = mmd_forward_run
    for e in enum_interesting_entries(qtype_entries, query_info_dict):
        info = query_info_dict[e.qid]

        input_ids = e.de_input_ids
        seg1_np, seg2_np = split_p_h_with_input_ids(input_ids, input_ids)
        seg2 = seg2_np.tolist()

        def tokenize(text):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        seg1 = tokenize(info.query)
        seg1_no_func = tokenize(info.content_span)

        if not len(seg1_no_func) < len(seg1):
            print("WARNING {} < {}".format(len(seg1_no_func), len(seg1)))
        base_score = forward_run(seg1, seg2)
        base_score_typeless = forward_run(seg1_no_func, seg2)

        single_sent_list = list(split_by_window(seg2, window_size))
        dropped_seg2s = list(enum_window_drop(seg2, unknown_token, window_size))

        sent_drop_result = [forward_run(seg1, window) for window in dropped_seg2s]
        sent_drop_result_typeless = [forward_run(seg1_no_func, window) for window in dropped_seg2s]

        single_sent_result = [forward_run(seg1, window) for window in single_sent_list]
        single_sent_result_typeless = [forward_run(seg1_no_func, window) for window in single_sent_list]

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


def print_base_info(e, query_info_dict):
    info = query_info_dict[e.qid]
    q_rep = " ".join(info.out_s_list)
    print(e.qid, q_rep)
    print("{}:{} - {}".format(e.doc_id, e.passage_idx, "Relevant" if e.label else "Non-relevant"))


def main():
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    run_contribution_analysis(qtype_entries, query_info_dict)


if __name__ == "__main__":
    main()
