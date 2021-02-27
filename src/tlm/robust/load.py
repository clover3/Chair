import os
import pickle
from typing import Dict, List

from data_generator.job_runner import sydney_working_dir
from list_lib import left

robust_query_intervals = [(301, 350), (351, 400), (401, 450), (601, 650), (651, 700)]


def get_robust_qid_list() -> List[int]:
    output = []
    for st, ed in robust_query_intervals:
        for qid in range(st, ed+1):
            output.append(qid)
    return output


robust_chunk_num = 33


def load_robust_tokens_for_train() -> Dict[str, List[str]]:
    data = {}
    for i in range(robust_chunk_num):
        path = os.path.join(sydney_working_dir, "RobustTokensClean", str(i))
        d = pickle.load(open(path, "rb"))
        if d is not None:
            data.update(d)
    return data


def load_robust_tokens_for_predict(version=3) -> Dict[str, List[str]]:
    data_name = "RobustPredictTokens{}".format(version)
    path = os.path.join(sydney_working_dir, data_name, "1")
    data = pickle.load(open(path, "rb"))
    return data


def load_robust_tokens() -> Dict[str, List[str]]:
    tokens_d = load_robust_tokens_for_train()
    tokens_d.update(load_robust_tokens_for_predict(4))
    return tokens_d


def get_robust_splits(split_idx):
    interval_start_list = left(robust_query_intervals)
    held_out = interval_start_list[split_idx]
    train_items = interval_start_list[:split_idx] + interval_start_list[split_idx + 1:]
    return train_items, held_out