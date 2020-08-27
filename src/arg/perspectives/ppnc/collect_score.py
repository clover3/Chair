import json
from typing import Dict, Tuple

from arg.perspectives.types import DataID, CPIDPair
from misc_lib import get_dir_files
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def collect_scores(prediction_file, info: Dict) \
        -> Dict[DataID, Tuple[CPIDPair, float]]:
    data = EstimatorPredictionViewer(prediction_file)
    print("Num data ", data.data_len)
    out_d: Dict[DataID, Tuple[CPIDPair, float]] = {}
    for entry in data:
        logits = entry.get_vector("logits")[0]
        data_id = entry.get_vector("data_id")[0]
        try:
            cur_info = info[str(data_id)]
            cid = cur_info['cid']
            pid = cur_info['pid']
            cpid = CPIDPair((cid, pid))
            out_d[data_id] = (cpid, logits)
        except KeyError as e:
            print("Key error")
            print("data_id", data_id)
            pass
    return out_d


def load_combine_info_jsons(dir_path) -> Dict:
    d = {}
    for file_path in get_dir_files(dir_path):
        if file_path.endswith(".info"):
            j = json.load(open(file_path, "r"))
            d.update(j)
    return d
