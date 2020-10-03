import json
import os
from typing import Dict, List

from arg.qck.decl import KDP, QCKQuery, QCKCandidate
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import dict_value_map
from misc_lib import get_dir_files, group_by

qck_convert_map = {
        'kdp': KDP,
        'query': QCKQuery,
        'candidate': QCKCandidate
    }
qk_convert_map = {
        'kdp': KDP,
        'query': QCKQuery,
    }

qc_convert_map = {
        'query': QCKQuery,
        'candidate': QCKCandidate,
    }

def load_combine_info_jsons(dir_path, convert_map, drop_kdp=True) -> Dict:
    if os.path.isdir(dir_path):
        d = {}
        for file_path in get_dir_files(dir_path):
            if file_path.endswith(".info"):
                j = json.load(open(file_path, "r", encoding="utf-8"))
                parse_info(j, convert_map, drop_kdp)
                d.update(j)
    else:
        d = json.load(open(dir_path, "r"))
        parse_info(d, convert_map, drop_kdp)
    return d
    # field


def parse_info(j, convert_map, drop_kdp):
    for data_id, info in j.items():
        parse_info_inner(info, convert_map, drop_kdp)


def parse_info_inner(info, convert_map, drop_kdp):
    for key, class_ in convert_map.items():
        if drop_kdp and key == "kdp" and len(info[key]) > 3:
            info[key][3] = []
        info[key] = class_(*info[key])


def load_prediction_with_info(info_path, pred_path, fetch_field_list=None) -> List[Dict]:
    info = load_combine_info_jsons(info_path, qck_convert_map)
    return join_prediction_with_info(pred_path, info, fetch_field_list)


def group_by_qid_cid(predictions: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    grouped: Dict[str, List[Dict]] = group_by(predictions, lambda x: x['query'].query_id)
    grouped2: Dict[str, Dict[str, List[Dict]]] = \
        dict_value_map(lambda x: group_by(x, lambda x: x['candidate'].id), grouped)
    return grouped2