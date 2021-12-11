from typing import Dict, List

import nltk
import numpy as np

from bert_api.client_lib import BERTClient
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.data_gen.doc_encode_common import join_tokens


def iter_de_tokens(records):
    max_seq_length = 512
    tokenizer = get_tokenizer()
    client = BERTClient("http://localhost", 8126, max_seq_length)

    def sent_payload(payload_np_list):
        def conv(t):
            out_l = []
            for e in list(t):
                l = e.tolist()
                assert type(l) == list
                assert type(l[0]) == int
                out_l.append(l)
            return tuple(out_l)
        payload_list = list(map(conv, payload_np_list))
        return client.send_payload(payload_list)

    def split_input_ids(input_ids):
        seg1, seg2 = split_p_h_with_input_ids(input_ids, input_ids)
        tokens1 = tokenizer.convert_ids_to_tokens(seg1)
        tokens2 = tokenizer.convert_ids_to_tokens(seg2)
        return tokens1, tokens2

    for record_idx, r in enumerate(records):
        if record_idx < 50:
            continue
        def get_field(name):
            return np.array(r[name].int64_list.value)

        qe_input_ids = get_field('q_e_input_ids')
        qe_input_mask = np.not_equal(qe_input_ids, 0).astype(np.int32)
        qe_segment_ids = get_field('q_e_segment_ids')
        de_input_ids = get_field('d_e_input_ids')
        de_input_mask = np.not_equal(de_input_ids, 0).astype(np.int32)
        de_segment_ids = get_field('d_e_segment_ids')

        payload_list = []
        orig_inst = qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids
        payload_list.append(orig_inst)
        content_tokens_, query_tokens = split_input_ids(qe_input_ids)
        content_tokens, document_tokens = split_input_ids(de_input_ids)

        if "proportion" in content_tokens:
            continue

        text = " ".join(document_tokens)
        sents = nltk.sent_tokenize(text)
        def build_payload(s):
            new_doc_tokens = tokenizer.tokenize(s)
            joined_tokens, new_segment_ids = join_tokens(content_tokens, new_doc_tokens)
            new_input_ids = tokenizer.convert_tokens_to_ids(joined_tokens)[:max_seq_length]
            new_segment_ids = new_segment_ids[:max_seq_length]
            new_input_mask = np.ones_like(new_input_ids, np.int)
            pad_len = max_seq_length - len(new_input_ids)
            pad_zeros = np.ones([pad_len], np.int)
            new_input_ids = np.concatenate([new_input_ids, pad_zeros])
            new_input_mask = np.concatenate([new_input_mask, pad_zeros])
            new_segment_ids = np.concatenate([new_segment_ids, pad_zeros])

            assert len(new_input_ids) == max_seq_length
            assert len(new_input_mask) == max_seq_length
            assert len(new_segment_ids) == max_seq_length
            one_inst = qe_input_ids, qe_input_mask, qe_segment_ids, new_input_ids, new_input_mask, new_segment_ids
            return one_inst

        orig_inst_res: Dict = sent_payload([orig_inst])[0]
        perturb_payloads = [build_payload(s) for s in sents]
        if orig_inst_res['logits'] < 2:
            continue
        print("Record #{}".format(record_idx))
        print("Send payload...")
        prediction_list: List[Dict] = sent_payload(perturb_payloads)
        print("DONE")

        q_sign = np.sign(orig_inst_res['qtype_vector1'])
        q_sign_normed = orig_inst_res['qtype_vector1'] * q_sign
        scale = np.max(q_sign_normed)
        # scale = 1/100
        def get_de_weights(inst):
            return np.array(inst['qtype_vector2']) * q_sign * scale

        orig_qe_weights = np.array(q_sign_normed) / scale
        orig_de_weights = get_de_weights(orig_inst_res)

        def show_min_k(weights):
            rank = np.argsort(weights)
            out_s = []
            for r in rank[:5]:
                out_s.append("{0} ({1:.2f})".format(r, weights[r]))
            return " ".join(out_s)

        def show_top_k(weights):
            rank = np.argsort(weights)[::-1]
            out_s = []
            for r in rank[:5]:
                out_s.append("{0} ({1:.2f})".format(r, weights[r]))
            return " ".join(out_s)

        print()
        print("Query: ", query_tokens)
        print("max_dim:", show_top_k(orig_qe_weights))
        print("min_dims", show_min_k(orig_qe_weights))
        print("Score d_bias q_bias qd_score")
        qd_score = np.dot(orig_inst_res["qtype_vector1"], orig_inst_res["qtype_vector2"])
        print("{0:.2f} {1:.2f} {2:.2f} {3:.2f}".
              format(orig_inst_res["logits"], orig_inst_res["d_bias"], orig_inst_res["q_bias"], qd_score))
        print("Orig:", text)
        print(show_top_k(orig_de_weights))
        print(show_min_k(orig_de_weights))

        for sent, res in zip(sents, prediction_list):
            de_weights = get_de_weights(res)
            print("sent: ", sent)
            print(show_top_k(de_weights))
            print(show_min_k(de_weights))