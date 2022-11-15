from port_info import KERAS_NLI_PORT, LOCAL_DECISION_PORT
from trainer_v2.custom_loop.definitions import ModelConfig300_3, ModelConfig600_3
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

