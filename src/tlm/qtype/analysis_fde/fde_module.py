from collections import defaultdict
from typing import Dict, List

import numpy as np

from bert_api.client_lib import BERTClient
from cpath import data_path, pjoin
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
        self.func_span_list: List[str] = func_span_list
        self.qtype_embedding_np = qtype_embedding_np
        self.q_bias_list = np.stack([q_bias_d[s] for s in func_span_list])

    def request(self, seg1_input_ids, seg2_input_ids):
        def flat(d):
            return d["input_ids"], d["input_mask"], d["segment_ids"]

        qe_input_ids, qe_input_mask, qe_segment_ids = self.q_encoder.encode_pair("", "")
        de_input_ids, de_input_mask, de_segment_ids = flat(self.d_encoder.encode_inner(seg1_input_ids, seg2_input_ids))
        one_inst = qe_input_ids, qe_input_mask, qe_segment_ids, de_input_ids, de_input_mask, de_segment_ids
        payload_list = [one_inst]
        ret: Dict = self.client.send_payload(payload_list)[0]
        assert type(ret) == dict
        return ret

    def compute_score(self, ct: str, doc: str):
        def encode(s):
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))
        return self.compute_score_from_ids(encode(ct), encode(doc))

    def compute_score_from_ids(self, ct: List[int], doc: List[int]) -> np.array:
        ret: Dict = self.request(ct, doc)
        doc_vector = ret['qtype_vector2']
        d_bias = ret['d_bias']
        score: np.array = np.dot(self.qtype_embedding_np, doc_vector) + d_bias + self.q_bias_list
        return score

    def get_promising_from_ids(self, ct: List[int], doc: List[int], threshold=0.5) -> List[str]:
        scores: np.array = self.compute_score_from_ids(ct, doc)
        output = self._filter_promising(scores, threshold)
        return output

    def get_promising(self, ct, doc, threshold=0.5) -> List[str]:
        scores = self.compute_score(ct, doc)
        output = self._filter_promising(scores, threshold)
        return output

    def _filter_promising(self, scores, threshold) -> List[str]:
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
    q_bias_d: Dict[str, np.array] = load_q_bias(run_name)
    return FDEModule(q_embedding_d, q_bias_d)


class FDEModuleEx(FDEModule):
    def __init__(self,
                 q_embedding_d: Dict[str, np.array],
                 q_bias_d: Dict[str, np.array],
                 cluster: List[int],
                 ):
        super(FDEModuleEx, self).__init__(q_embedding_d, q_bias_d)
        self.cluster = cluster
        cluster_id_to_idx = defaultdict(list)
        func_span_to_cluster_idx = {}
        for idx, cluster_id in enumerate(cluster):
            func_span = self.func_span_list[idx]
            cluster_id_to_idx[cluster_id].append(idx)
            func_span_to_cluster_idx[func_span] = cluster_id
        self.cluster_id_to_idx = cluster_id_to_idx
        self.func_span_to_cluster_idx = func_span_to_cluster_idx

    def get_cluster_id(self, func_span):
        return self.func_span_to_cluster_idx[func_span]
