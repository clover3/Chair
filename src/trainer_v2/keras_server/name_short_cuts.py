from typing import List, Callable, Tuple

from bert_api.task_clients.nli_interface.nli_predictors_path import get_pep_cache_sqlite_path, \
    get_nli14_cache_sqlite_path
from datastore.sql_based_cache_client import SQLBasedCacheClientS
from port_info import KERAS_NLI_PORT, LOCAL_DECISION_PORT
from trainer_v2.custom_loop.definitions import ModelConfig300_3, ModelConfig600_3
from trainer_v2.keras_server.bert_like_client import BERTClient

NLIPredictorSig = Callable[[List[Tuple[str, str]]], List[List[float]]]


def get_nli14_client():
    model_config = ModelConfig300_3()
    client = BERTClient("localhost", KERAS_NLI_PORT, model_config.max_seq_length)
    return client


def get_nli14_predictor() -> NLIPredictorSig:
    client = get_nli14_client()
    return client.request_multiple


def get_pep_client() -> NLIPredictorSig:
    model_config = ModelConfig600_3()
    client = BERTClient("localhost", LOCAL_DECISION_PORT, model_config.max_seq_length)

    def predict(items) -> List[List[float]]:
        result = client.request_multiple(items)
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
    forward_fn_raw: NLIPredictorSig = get_nli14_predictor()
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
