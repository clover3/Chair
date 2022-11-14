import os
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
from cpath import data_path

from data_generator2.encoder_unit import EncoderUnitK
from utils.xml_rpc_helper import ServerProxyEx

KERAS_BERT_LIKE_INPUT_SIG = List[Tuple[List[int], List[int]]]


class BERTClient:
    def __init__(self, server_addr, port, seq_len=512):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = EncoderUnitK(seq_len, voca_path)
        self.proxy = ServerProxyEx(server_addr, port)

    def request_single(self, text1, text2):
        return self.proxy.send([self.encoder.encode_pair(text1, text2)])

    def request_multiple(self, text_pair_list: List[Tuple[str, str]]):
        payload = []
        for text1, text2 in text_pair_list:
            payload.append(self.encoder.encode_pair(text1, text2))
        return self.proxy.send(payload)

    def request_multiple_from_tokens(self, payload_list: List[Tuple[List[str], List[str]]]):
        conv_payload_list = []
        for tokens_a, tokens_b in payload_list:
            conv_payload_list.append(self.encoder.encode_token_pairs(tokens_a, tokens_b))
        return self.proxy.send(conv_payload_list)
