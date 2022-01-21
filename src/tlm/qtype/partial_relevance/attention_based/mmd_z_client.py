from typing import List, Callable

import numpy as np
from scipy.special import softmax

from bert_api.client_lib import BERTClient
from data_generator.tokenizer_wo_tf import JoinEncoder
from port_info import MMD_Z_PORT
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance


def get_mmd_client_wrap() -> Callable[[List[SegmentedInstance]], List[float]]:
    max_seq_length = 512
    client = BERTClient("http://localhost", MMD_Z_PORT, max_seq_length)
    join_encoder = JoinEncoder(max_seq_length)

    def query_multiple(items: List[SegmentedInstance]) -> List[float]:
        if len(items) == 0:
            return []
        def encode(item: SegmentedInstance):
            return join_encoder.join(item.text1.tokens_ids, item.text2.tokens_ids)
        ret = client.send_payload(list(map(encode, items)))
        ret = np.array(ret)
        probs = softmax(ret, axis=1)[:, 1]
        return probs
    return query_multiple

