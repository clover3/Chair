import os
import pickle
from typing import Dict, List

from data_generator.job_runner import sydney_working_dir

robust_query_intervals = [(301, 350), (351, 400), (401, 450), (601, 650), (651, 700)]

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
    print(data_name)
    path = os.path.join(sydney_working_dir, data_name, "1")
    data = pickle.load(open(path, "rb"))
    return data

