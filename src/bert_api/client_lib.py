import os
import xmlrpc.client
from typing import List, Tuple

from bert_api.msmarco_tokenization import EncoderUnit
from cpath import data_path


class BERTClient:
    def __init__(self, server_addr, port, seq_len=512):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = EncoderUnit(seq_len, voca_path)
        self.proxy = xmlrpc.client.ServerProxy('{}:{}'.format(server_addr, port))

    def request_single(self, text1, text2):
        payload = []
        payload.append(self.encoder.encode_pair(text1, text2))
        return self.send_payload(payload)

    def send_payload(self, payload):
        if payload:
            r = self.proxy.predict(payload)
            return r
        else:
            return []

    def request_multiple(self, text_pair_list: List[Tuple[str, str]]):
        payload = []
        for text1, text2 in text_pair_list:
            payload.append(self.encoder.encode_pair(text1, text2))
        return self.send_payload(payload)

    def request_multiple_from_tokens(self, payload_list: List[Tuple[List[str], List[str]]]):
        conv_payload_list = []
        for tokens_a, tokens_b in payload_list:
            conv_payload_list.append(self.encoder.encode_token_pairs(tokens_a, tokens_b))
        return self.send_payload(conv_payload_list)


def get_ingham_client():
    server_addr = "http://ingham.cs.umass.edu"
    port = 8122
    return BERTClient(server_addr, port)


def get_localhost_client():
    server_addr = "http://localhost"
    port = 8122
    return BERTClient(server_addr, port)


def simple_test():
    text = "Practical text classification system should be able to utilize information from both expensive labelled documents and large volumes of cheap unlabelled documents. It should also easily deal with newly input samples. In this paper, we propose a random walks method for text classification, in which the classification problem is formulated as solving the absorption probabilities of Markov random walks on a weighted graph. Then the Laplacian operator for asymmetric graphs is derived and utilized for asymmetric transition matrix. We also develop an induction algorithm for the newly input documents based on the random walks method. Meanwhile, to make full use of text information, a difference measure for text data based on language model and KL-divergence is proposed, as well as a new smoothing technique for it. Finally an algorithm for elimination of ambiguous states is proposed to address the problem of noisy data. Experiments on two well-known data sets: W ebKB and 20Newsgroup demonstrate the effectivity of the proposed random walks method."
    query = "A Random Walks Method for Text Classification"
    client = get_ingham_client()
    r = client.request_single(text, query)
    print(r)


if __name__ == "__main__":
    simple_test()
