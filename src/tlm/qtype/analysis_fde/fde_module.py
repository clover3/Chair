import os
from typing import Dict

import numpy as np

from bert_api.client_lib import BERTClient
from cache import load_pickle_from
from cpath import data_path, pjoin
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, EncoderUnitPlain
from port_info import FDE_PORT
from tlm.qtype.analysis_fde.analysis_a import embeddings_to_list
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_bias
from tlm.qtype.analysis_fde.runner.build_q_emb_verify import load_q_emb_qtype_2Y_v_train_120000
from trainer.np_modules import sigmoid


class FDEModule:
    def __init__(self,
                 q_embedding_d: Dict[str, np.array],
                 q_bias_d: Dict[str, np.array],
                 ):
        max_seq_length = 512
        self.client = BERTClient("http://localhost", FDE_PORT, max_seq_length)
        self.tokenizer = get_tokenizer()
        voca_path = pjoin(data_path, "bert_voca.txt")
        self.q_encoder = EncoderUnitPlain(128, voca_path)
        self.d_encoder = EncoderUnitPlain(max_seq_length, voca_path)
        self.q_embedding_d = q_embedding_d
        self.q_bias_d = q_bias_d
        func_span_list, qtype_embedding_np = embeddings_to_list(q_embedding_d)
        self.func_span_list = func_span_list
        print(len(func_span_list), len(q_bias_d))
        self.qtype_embedding_np = qtype_embedding_np
        self.q_bias_list = np.stack([q_bias_d[s] for s in func_span_list])

    def request(self, seg1_input_ids, seg2_input_ids):
        def flat(d):
            return d["input_ids"], d["input_mask"], d["segment_ids"]

        qe_input_ids, qe_input_mask, qe_segment_ids = self.q_encoder.encode_pair("", "")
        de_input_ids, de_input_mask, de_segment_ids = flat(self.d_encoder.encode_inner(seg1_input_ids, seg2_input_ids))
        one_inst = qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids
        payload_list = [one_inst]
        ret = self.client.send_payload(payload_list)[0]
        return ret

    def compute_score(self, ct: str, doc: str):
        def encode(s):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))

        ret = self.request(encode(ct), encode(doc))
        doc_vector = ret['qtype_vector2']
        d_bias = ret['d_bias']
        score = np.dot(self.qtype_embedding_np, doc_vector) + d_bias + self.q_bias_list
        return score

    def get_promising(self, ct, doc, threshold=0.5):
        scores = self.compute_score(ct, doc)
        probs = sigmoid(scores)

        relevant = np.less(threshold, probs)
        output = []
        if np.count_nonzero(relevant):
            for i in range(len(probs)):
                if relevant[i]:
                    output.append(self.func_span_list[i])
        return output


def get_fde_module() -> FDEModule:
    run_name = "qtype_2Y_v_train_120000"
    q_embedding_d: Dict[str, np.array] = load_q_emb_qtype_2Y_v_train_120000()
    save_dir = os.path.join(output_path, "qtype", run_name + '_sample')
    _, query_info_dict = load_pickle_from(os.path.join(save_dir, "0"))
    q_bias_d: Dict[str, np.array] = load_q_bias(run_name)
    return FDEModule(q_embedding_d, q_bias_d)
