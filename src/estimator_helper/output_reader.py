import json
import os
from typing import Dict, Any, List

from misc_lib import get_dir_files
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def load_combine_info_jsons(dir_path) -> Dict:
    if os.path.isdir(dir_path):
        d = {}
        for file_path in get_dir_files(dir_path):
            if file_path.endswith(".info"):
                j = json.load(open(file_path, "r", encoding="utf-8"))
                d.update(j)
    else:
        d = json.load(open(dir_path, "r"))
    print("{} items loaded".format(len(d.keys())))
    return d


def join_prediction_with_info(prediction_file,
                              info: Dict[str, Any],
                              fetch_field_list=None,
                              str_data_id=True
                              ) -> List[Dict]:
    if fetch_field_list is None:
        fetch_field_list = ["logits", "data_id"]
    print("Reading pickle...")
    data = EstimatorPredictionViewer(prediction_file, fetch_field_list)
    print("Num data ", data.data_len)
    seen_data_id = set()
    out = []
    not_found_cnt = 0
    for entry in data:
        data_id = entry.get_vector("data_id")[0]
        try:
            if str_data_id:
                k_data_id = str(data_id)
            else:
                k_data_id = data_id

            cur_info = info[k_data_id]
            assert data_id not in seen_data_id
            seen_data_id.add(data_id)
            new_entry = dict(cur_info)
            for field in fetch_field_list:
                new_entry[field] = entry.get_vector(field)
            out.append(new_entry)
        except KeyError as e:
            print(e)
            print("Key error", e.__str__())
            print("data_id", data_id)
            not_found_cnt += 1
            if not_found_cnt > 100:
                raise ReferenceError()
            pass
    return out



# field
def load_prediction_with_info(info_path, pred_path) -> List[Dict]:
    info = load_combine_info_jsons(info_path)
    return join_prediction_with_info(pred_path, info)

