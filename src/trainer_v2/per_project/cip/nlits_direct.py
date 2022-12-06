import os
from typing import List, Tuple

import cpath
from data_generator2.encoder_unit import EncoderUnitK
from trainer_v2.custom_loop.definitions import ModelConfig600_3


def reslice_local_global_decisions(stacked_output):
    l_decisions, g_decision_l = stacked_output
    n_item = len(l_decisions)
    assert len(l_decisions) == len(g_decision_l[0])
    output = []
    for i in range(n_item):
        output.append((l_decisions[i].tolist(), g_decision_l[0][i].tolist()))
    return output


class TS600_3_Encoder:
    def __init__(self):
        model_config = ModelConfig600_3()
        self.segment_len = int(model_config.max_seq_length / 2)
        voca_path = os.path.join(cpath.data_path, "bert_voca.txt")
        self.encoder = EncoderUnitK(self.segment_len, voca_path)

    def combine_ts_triplets(self, payload_list: List[Tuple[List[int], List[int], List[int]]]):
        conv_payload_list = []
        for p_ids, h_ids1, h_ids2 in payload_list:
            input_ids1, segment_ids1 = self.encoder.encode_from_ids(p_ids, h_ids1)
            input_ids2, segment_ids2 = self.encoder.encode_from_ids(p_ids, h_ids2)
            input_ids = input_ids1 + input_ids2
            segment_ids = segment_ids1 + segment_ids2
            conv_payload_list.append((input_ids, segment_ids))
        return conv_payload_list

