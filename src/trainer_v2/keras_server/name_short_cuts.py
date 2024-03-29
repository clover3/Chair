from typing import List, Callable, Tuple

from bert_api.task_clients.nli_interface.nli_predictors_path import get_pep_cache_sqlite_path, \
    get_nli14_cache_sqlite_path
from cpath import get_canonical_model_path2
from datastore.sql_based_cache_client import SQLBasedCacheClientS
from port_info import KERAS_NLI_PORT, LOCAL_DECISION_PORT
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.attention_helper.model_shortcut import nli14_model_path
from trainer_v2.custom_loop.definitions import ModelConfig300_3, ModelConfig600_3
from trainer_v2.keras_server.bert_like_client import BERTClient, BERTClientCore
from trainer_v2.keras_server.bert_like_server import get_keras_bert_like_predict_fn

IDS = List[int]
NLIPredictorSig = Callable[[List[Tuple[str, str]]], List[List[float]]]


def get_keras_nli_300_client():
    model_config = ModelConfig300_3()
    client = BERTClient("localhost", KERAS_NLI_PORT, model_config.max_seq_length)
    return client


def get_keras_nli_300_predictor() -> NLIPredictorSig:
    client = get_keras_nli_300_client()
    return client.request_multiple


def tokenize_w_mask_preserving(full_tokenizer, text):
    split_tokens = []
    for sp_tokens in text.split():
        if sp_tokens == "[MASK]":
            split_tokens.append(sp_tokens)
        else:
            for token in full_tokenizer.basic_tokenizer.tokenize(sp_tokens):
                for sub_token in full_tokenizer.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
    return split_tokens


def get_pep_client(verbose=False) -> NLIPredictorSig:
    model_config = ModelConfig600_3()
    client = BERTClient("localhost", LOCAL_DECISION_PORT, model_config.max_seq_length)
    full_tokenizer = client.encoder.encoder.ft

    def encode_tuple(t: Tuple[str, str]) -> Tuple[IDS, IDS]:
        def encode_one(s: str) -> IDS:
            tokens = tokenize_w_mask_preserving(full_tokenizer, s)
            return full_tokenizer.convert_tokens_to_ids(tokens)

        s1, s2 = t
        return encode_one(s1), encode_one(s2)

    def predict(items: List[Tuple[str, str]]) -> List[List[float]]:
        c_log.debug("Encoding items...")
        tokenized: List[Tuple[IDS, IDS]] = list(map(encode_tuple, items))
        c_log.debug("Sending items...")
        result = client.request_multiple_from_ids_pairs(tokenized)
        c_log.debug("Received.")

        output = []
        for local_decision, g_decision in result:
            output.append(local_decision[0])
        return output

    return predict


def tuple_to_str(s_tuple: Tuple[str, str]) -> str:
    s1, s2 = s_tuple
    return s1 + " [SEP] " + s2


def get_pep_cache_client(hooking_fn=None) -> NLIPredictorSig:
    forward_fn_raw: NLIPredictorSig = get_pep_client()
    sqlite_path = get_pep_cache_sqlite_path()
    cache_client = get_cached_client(forward_fn_raw, hooking_fn, sqlite_path)
    return cache_client.predict


def get_nli14_cache_client(hooking_fn=None) -> NLIPredictorSig:
    forward_fn_raw: NLIPredictorSig = get_keras_nli_300_predictor()
    sqlite_path = get_nli14_cache_sqlite_path()
    cache_client = get_cached_client(forward_fn_raw, hooking_fn, sqlite_path)
    return cache_client.predict


def get_cached_client(forward_fn_raw, hooking_fn, sqlite_path):
    if hooking_fn is not None:
        def forward_fn(items: List) -> List[List[float]]:
            hooking_fn(items)
            return forward_fn_raw(items)
    else:
        forward_fn = forward_fn_raw
    cache_client = SQLBasedCacheClientS(forward_fn,
                                        tuple_to_str,
                                        0.035,
                                        sqlite_path)
    return cache_client


def get_nli14_direct(strategy):
    model_path = nli14_model_path()
    model_config = ModelConfig300_3()
    predict_fn = get_keras_bert_like_predict_fn(model_path, model_config, strategy)
    return BERTClientCore(predict_fn, model_config.max_seq_length).request_multiple

