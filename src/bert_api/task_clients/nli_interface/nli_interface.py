from typing import List, Callable, NamedTuple

import scipy.special

from bert_api.client_lib import BERTClient
from bert_api.segmented_instance.segmented_text import SegmentedText, get_word_level_segmented_text_from_str
from cpath import pjoin, data_path, at_output_dir
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from datastore.sql_based_cache_client import SQLBasedCacheClientS
from port_info import NLI_PORT


class NLIInput(NamedTuple):
    prem: SegmentedText
    hypo: SegmentedText

    def str_hash(self):
        return str(self.prem.tokens_ids) + str(self.hypo.tokens_ids)


NLIPredictorSig = Callable[[List[NLIInput]], List[List[float]]]


def get_nli_client_by_server() -> NLIPredictorSig:
    max_seq_length = 300
    client = BERTClient("http://localhost", NLI_PORT, max_seq_length)
    voca_path = pjoin(data_path, "bert_voca.txt")
    d_encoder = EncoderUnitPlain(max_seq_length, voca_path)
    def query_multiple(input_list: List[NLIInput]):
        payload = []
        for nli_input in input_list:
            p_tokens_id: List[int] = nli_input.prem.tokens_ids
            h_tokens_id: List[int] = nli_input.hypo.tokens_ids
            d = d_encoder.encode_inner(p_tokens_id, h_tokens_id)
            p = d["input_ids"], d["input_mask"], d["segment_ids"]
            payload.append(p)
        ret = client.send_payload(payload)
        return ret
    return query_multiple


def get_nli_client(option: str) -> NLIPredictorSig:
    if option == "localhost":
        return get_nli_client_by_server()
    elif option == "direct":
        print("use direct predictor")
        return NotImplemented
    else:
        raise ValueError


def get_nli_cache_sqlite_path():
    return at_output_dir("nli", "nli_cache.sqlite")


def get_nli_cache_client(option, hooking_fn=None) -> SQLBasedCacheClientS:
    forward_fn_raw: NLIPredictorSig = get_nli_client(option)
    if hooking_fn is not None:
        def forward_fn(items: List[NLIInput]) -> List[List[float]]:
            hooking_fn(items)
            return forward_fn_raw(items)
    else:
        forward_fn = forward_fn_raw

    cache_client = SQLBasedCacheClientS(forward_fn,
                                        NLIInput.str_hash,
                                        0.035,
                                        get_nli_cache_sqlite_path())
    return cache_client



def predict_from_text_pair(client: SQLBasedCacheClientS, tokenizer, text1, text2):
    t_text1: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, text1)
    t_text2: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, text2)

    logits = client.predict([NLIInput(t_text1, t_text2)])[0]
    probs = scipy.special.softmax(logits)
    return probs
