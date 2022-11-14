from port_info import KERAS_NLI_PORT
from trainer_v2.custom_loop.definitions import ModelConfig300_3
from trainer_v2.keras_server.bert_like_client import BERTClient
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple

NLIPredictorSig = Callable[[List[Tuple[str, str]]], List[List[float]]]


def get_nli14_client():
    model_config = ModelConfig300_3()
    client = BERTClient("localhost", KERAS_NLI_PORT, model_config.max_seq_length)
    return client


def get_nli14_predictor() -> NLIPredictorSig:
    client = get_nli14_client()
    return client.request_multiple
