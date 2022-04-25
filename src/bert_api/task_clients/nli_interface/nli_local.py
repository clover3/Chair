from bert_api.task_clients.nli_interface.nli_interface import NLIPredictorSig, NLIInput
from contradiction.ie_align.srl.tool.get_nli_predictor_common import load_standard_nli_model
from cpath import data_path, pjoin
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from typing import List, Iterable, Callable, Dict, Tuple, Set
import tensorflow as tf
import numpy as np
from trainer.np_modules import get_batches_ex


def get_local_nli_client() -> NLIPredictorSig:
    max_seq_length = 300
    voca_path = pjoin(data_path, "bert_voca.txt")
    batch_size = 16
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)
    bert_classifier_layer = load_standard_nli_model()
    def query_multiple(input_list: List[NLIInput]) -> List[List[float]]:
        payload = []
        for nli_input in input_list:
            p_tokens_id: List[int] = nli_input.prem.tokens_ids
            h_tokens_id: List[int] = nli_input.hypo.tokens_ids
            d = d_encoder.encode_inner(p_tokens_id, h_tokens_id)
            p = d["input_ids"], d["input_mask"], d["segment_ids"]
            payload.append(p)
        batches = get_batches_ex(payload, batch_size, 3)

        probs_list = []
        for batch in batches:
            x0, x1, x2 = batch
            logits = bert_classifier_layer((x0, x1, x2))
            probs = tf.nn.softmax(logits)
            probs_list.append(probs.numpy())
        return np.concatenate(probs_list, 0).tolist()
    return query_multiple