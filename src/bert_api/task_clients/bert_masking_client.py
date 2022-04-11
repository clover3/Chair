import xmlrpc.client
from typing import List, Tuple

from port_info import BERT_MASK_PORT
from bert_api.bert_masking_common import ATTENTION_MASK, SinglePayload, \
    serialize_tuple_d, BERTMaskIF

SEG1 = List[List[int]]
SEG2 = List[List[int]]
SingleInput = Tuple[SEG1, SEG2, ATTENTION_MASK]


class BERTMaskClient(BERTMaskIF):
    def __init__(self, server_addr, port):
        self.proxy = xmlrpc.client.ServerProxy('{}:{}'.format(server_addr, port))

    def send_payload(self, payload: List[SinglePayload]):
        if payload:
            payload = serialize_tuple_d(payload)
            r = self.proxy.predict(payload)
            return r
        else:
            return []

    def predict(self, items: List[SinglePayload]):
        return self.send_payload(items)


def get_localhost_bert_mask_client() -> BERTMaskClient:
    client = BERTMaskClient("http://localhost", BERT_MASK_PORT)
    return client


