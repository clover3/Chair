from typing import List

from cache import load_list_from_jsonl, save_list_to_jsonl_w_fn
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction


def save_sent_token_label(labels: List[SentTokenLabel], save_path):
    return save_list_to_jsonl_w_fn(labels, save_path, SentTokenLabel.to_json)


def save_sent_token_binary_predictions(predictions: List[SentTokenBPrediction], save_path):
    return save_list_to_jsonl_w_fn(predictions, save_path, SentTokenBPrediction.to_json)


def load_sent_token_binary_predictions(save_path):
    return load_list_from_jsonl(save_path, SentTokenBPrediction.from_json)


def load_sent_token_label(file_path) -> List[SentTokenLabel]:
    label_list: List[SentTokenLabel] = load_list_from_jsonl(file_path, SentTokenLabel.from_json)
    return label_list


