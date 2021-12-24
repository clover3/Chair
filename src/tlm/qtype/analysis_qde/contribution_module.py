from typing import List

import numpy as np

from bert_api.client_lib import BERTClient
from cache import load_from_pickle
from cpath import QDE_PORT, data_path, pjoin
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import convert_ids_to_tokens, get_tokenizer, EncoderUnitPlain, pretty_tokens
from tlm.data_gen.doc_encode_common import split_by_window
from tlm.qtype.analysis_qemb.save_parsed import get_voca_list
from tlm.qtype.qtype_instance import QTypeInstance


def seg_contribution_from_scores(
        base_score: float,
        single_sent_result: List[float],
        sent_drop_result: List[float]):
    single_sent_based_score = [after for after in single_sent_result]
    drop_sent_based_score = contribution_by_drop(base_score, sent_drop_result)
    avg_sent_score = [(a+b)/2 for a, b in zip(single_sent_based_score, drop_sent_based_score)]
    return avg_sent_score


def contribution_by_drop(base_score, sent_drop_result):
    drop_sent_based_score = [base_score - after for after in sent_drop_result]
    return drop_sent_based_score


def enum_interesting_entries(qtype_entries: List[QTypeInstance], query_info_dict):
    # qtype_embedding_paired, n_query = build_qtype_desc(qtype_entries, query_info_dict)
    for e_idx, e in enumerate(qtype_entries):
        f_high_logit = e.logits > 3 or e.d_bias > 3

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


def enum_window_drop(tokens, unknown_token, window_size):
    cursor = 0
    while cursor < len(tokens):
        ed = cursor + window_size
        head = tokens[:cursor]
        tail = tokens[ed:]
        out_tokens = head + [unknown_token] + tail
        yield out_tokens
        cursor = ed


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


def run_contribution_analysis(qtype_entries, query_info_dict):
    tokenizer = get_tokenizer()
    voca_list = get_voca_list(tokenizer)
    unknown_token = tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
    window_size = 20
    qde_client = QDEClientWrap()
    for e in enum_interesting_entries(qtype_entries, query_info_dict):
        input_ids = e.de_input_ids
        seg1_np, seg2_np = split_p_h_with_input_ids(input_ids, input_ids)
        seg1 = seg1_np.tolist()
        seg2 = seg2_np.tolist()

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

        base_score = qde_forward_run(seg1, seg2)
        base_score_typeless = qde_forward_run_typeless(seg1, seg2)

        single_sent_list = list(split_by_window(seg2, window_size))
        dropped_seg2s = list(enum_window_drop(seg2, unknown_token, window_size))

        sent_drop_result = [qde_forward_run(seg1, window) for window in dropped_seg2s]
        sent_drop_result_typeless = [qde_forward_run_typeless(seg1, window) for window in dropped_seg2s]

        single_sent_result = [qde_forward_run(seg1, window) for window in single_sent_list]
        single_sent_result_typeless = [qde_forward_run_typeless(seg1, window) for window in single_sent_list]

        contribution_array = contribution_by_drop(base_score,
                                                          single_sent_result)
                                                          # sent_drop_result)
        contribution_array_typeless = contribution_by_drop(base_score_typeless,
                                                                   single_sent_result_typeless,)
                                                                   # sent_drop_result_typeless,
                                                                   # )

        info = query_info_dict[e.qid]
        q_rep = " ".join(info.out_s_list)
        print(e.qid, q_rep)
        print("{}:{} - {}".format(e.doc_id, e.passage_idx, "Relevant" if e.label else "Non-relevant"))
        print("Base score: {0:.2f}".format(base_score))
        base_gap = base_score - base_score_typeless
        for window_idx, window in enumerate(split_by_window(seg2, window_size)):
            seg2_tokens = convert_ids_to_tokens(voca_list, window)
            passage: str = pretty_tokens(seg2_tokens, True)
            relative_type_contribution = contribution_array[window_idx] - contribution_array_typeless[window_idx] - base_gap
            print("{0:.2f}\t{1:.2f}\t{2:.2f}\t{3}".format(
                contribution_array[window_idx],
                contribution_array_typeless[window_idx],
                relative_type_contribution,
                passage))

        print("")


def main():
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    run_contribution_analysis(qtype_entries, query_info_dict)


if __name__ == "__main__":
    main()
