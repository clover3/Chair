import os

from bert_api.predictor import Predictor
from bert_api.task_clients.nli_interface.nli_interface import NLIPredictorSig, NLIInput
from cpath import data_path, pjoin, output_path
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from typing import List, Iterable, Callable, Dict, Tuple, Set
import tensorflow as tf
import numpy as np
from trainer.np_modules import get_batches_ex


def get_local_nli_client() -> NLIPredictorSig:
    max_seq_length = 300
    voca_path = pjoin(data_path, "bert_voca.txt")
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)
    model_path = os.path.join(output_path, "model", "runs", "standard_nli", "model-73630")
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