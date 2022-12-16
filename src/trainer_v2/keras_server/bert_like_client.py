import os
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
from cpath import data_path

from data_generator2.encoder_unit import EncoderUnitK
from utils.xml_rpc_helper import ServerProxyEx

KERAS_BERT_LIKE_INPUT_SIG = List[Tuple[List[int], List[int]]]


class BERTClientCore:
    def __init__(self, proxy_predict, seq_len):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = EncoderUnitK(seq_len, voca_path)
        self.proxy_predict = proxy_predict

    def request_single(self, text1, text2):
        return self.proxy_predict([self.encoder.encode_pair(text1, text2)])

    def request_multiple(self, text_pair_list: List[Tuple[str, str]]):
        payload = []
        for text1, text2 in text_pair_list:
            payload.append(self.encoder.encode_pair(text1, text2))
        return self.proxy_predict(payload)

    def request_multiple_from_tokens(self, payload_list: List[Tuple[List[str], List[str]]]):
        conv_payload_list = []
        for tokens_a, tokens_b in payload_list:
            conv_payload_list.append(self.encoder.encode_token_pairs(tokens_a, tokens_b))
        return self.proxy_predict(conv_payload_list)

    def request_multiple_from_ids_pairs(self, payload_list: List[Tuple[List[int], List[int]]]):
        conv_payload_list = []
        for ids_a, ids_b in payload_list:
            conv_payload_list.append(self.encoder.encode_from_ids(ids_a, ids_b))
        return self.proxy_predict(conv_payload_list)


class BERTClient(BERTClientCore):
    def __init__(self, server_addr, port, seq_len=512):
        proxy = ServerProxyEx(server_addr, port)
        super(BERTClient, self).__init__(proxy.send, seq_len)
