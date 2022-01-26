import os
from typing import List, Callable

import numpy as np
from scipy.special import softmax

from cpath import output_path
from data_generator.tokenizer_wo_tf import JoinEncoder
from tf_v2_support import disable_eager_execution
from tlm.qtype.model_server.mmd_server import PredictorClsDense
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance


def mmd_predictor():
    save_path = os.path.join(output_path, "model", "runs", "mmd_Z")
    disable_eager_execution()

    predictor = PredictorClsDense(2, 512)
    load_names = ['bert', "cls_dense"]
    predictor.load_model_white(save_path, load_names)

    def predict(payload):
        sout = predictor.predict(payload)
        return sout
    return predict


def get_mmd_z_direct_wrap() -> Callable[[List[SegmentedInstance]], List[float]]:
    max_seq_length = 512
    join_encoder = JoinEncoder(max_seq_length)
    predict = mmd_predictor()
    def query_multiple(items: List[SegmentedInstance]) -> List[float]:
        if len(items) == 0:
            return []
        def encode(item: SegmentedInstance):
            return join_encoder.join(item.text1.tokens_ids, item.text2.tokens_ids)
        ret = predict(list(map(encode, items)))
        ret = np.array(ret)
        probs = softmax(ret, axis=1)[:, 1]
        return probs
    return query_multiple

