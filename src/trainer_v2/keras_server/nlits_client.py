import os
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
from cpath import data_path

from data_generator2.encoder_unit import EncoderUnitK
from utils.xml_rpc_helper import ServerProxyEx


class NLITSClient:
    def __init__(self, server_addr, port, segment_len):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = EncoderUnitK(segment_len, voca_path)
        self.proxy = ServerProxyEx(server_addr, port)

    def request_multiple_from_ids_triplets(
            self,
            payload_list: List[Tuple[List[int], List[int], List[int]]]):
        conv_payload_list = []
        for p_ids, h_ids1, h_ids2 in payload_list:
            input_ids1, segment_ids1 = self.encoder.encode_from_ids(p_ids, h_ids1)
            input_ids2, segment_ids2 = self.encoder.encode_from_ids(p_ids, h_ids2)
            input_ids = input_ids1 + input_ids2
            segment_ids = segment_ids1 + segment_ids2
            conv_payload_list.append((input_ids, segment_ids))
        return self.proxy.send(conv_payload_list)

