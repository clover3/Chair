import json
import os
from typing import Dict, Callable, Tuple, List

import scipy.special

from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import group_by, tprint, average, get_dir_files


def summarize_score(info: Dict,
                    prediction_file_path: str,
                    get_pair_id,
                    combine_score: Callable,
                    score_type) -> Dict[Tuple[str, str], float]:
    key_logit = "logits"
    data: List[Dict] = join_prediction_with_info(prediction_file_path, info, ["data_id", key_logit, "label_ids"])

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    def get_score(entry):
        if score_type == "softmax":
            return logit_to_score_softmax(entry['logits'])
        elif score_type == "raw":
            return entry[key_logit][0]
        elif score_type == "scalar":
            return entry[key_logit]
        elif score_type == "tuple":
            return entry[key_logit][1]
        elif score_type == "label_ids":
            return entry["label_ids"][0]
        else:
            assert False

    grouped: Dict[Tuple[str, str], List[Dict]] = group_by(data, get_pair_id)
    tprint("Group size:", len(grouped))
    out_d = {}
    for pair_id, items in grouped.items():
        scores = lmap(get_score, items)
        final_score = combine_score(scores)
        out_d[pair_id] = final_score

    num_items_per_group = average(lmap(len, grouped.values()))
    tprint("Num items per group : ", num_items_per_group)
    return out_d


def load_info_jsons(dir_path) -> Dict:
    if os.path.isdir(dir_path):
        d = {}
        for file_path in get_dir_files(dir_path):
            if file_path.endswith(".info"):
                j = json.load(open(file_path, "r", encoding="utf-8"))
                d.update(j)
    else:
        d = json.load(open(dir_path, "r"))
    return d
    # field


def get_pair_id(e):
    return e['qid'], e['data_id'][0]