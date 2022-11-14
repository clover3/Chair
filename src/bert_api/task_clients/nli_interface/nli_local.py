import os
from typing import List

from bert_api.task_clients.nli_interface.nli_interface import NLIPredictorFromSegTextSig, NLIInput
from cpath import data_path, pjoin, output_path
from data_generator.tokenizer_wo_tf import EncoderUnitPlain


# Return probabilities
def get_local_nli_client() -> NLIPredictorFromSegTextSig:
    from bert_api.predictor import Predictor

    max_seq_length = 300
    voca_path = pjoin(data_path, "bert_voca.txt")
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)
    model_path = os.path.join(output_path, "model", "runs", "standard_nli")
    inner_predictor = Predictor(model_path, 3, max_seq_length, False)
    def query_multiple(input_list: List[NLIInput]) -> List[List]:
        payload = []
        for nli_input in input_list:
            p_tokens_id: List[int] = nli_input.prem.tokens_ids
            h_tokens_id: List[int] = nli_input.hypo.tokens_ids
            d = d_encoder.encode_inner(p_tokens_id, h_tokens_id)
            p = d["input_ids"], d["input_mask"], d["segment_ids"]
            payload.append(p)
        return inner_predictor.predict(payload)
    return query_multiple