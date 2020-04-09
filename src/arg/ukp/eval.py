import pickle
from typing import Dict, List, Tuple

import numpy as np

from arg.pf_common.base import DPID
from arg.ukp.data_loader import load_all_data, UkpDataPoint
from arg.ukp.ukp_para_eval import get_datapoint_score
from base_type import FileName, FilePath
from cpath import output_path, pjoin
from data_generator.argmining.ukp_header import label_names
from list_lib import dict_value_map, lmap, right
from task.metrics import eval_3label, eval_2label


def get_dev_labels(topic) -> List[Tuple[DPID, int]]:
    _, raw_data = load_all_data()

    data: List[UkpDataPoint] = raw_data[topic]

    def simplify(e: UkpDataPoint) -> Tuple[DPID, int]:
        return DPID(str(e.id)), label_names.index(e.label)

    return lmap(simplify, data)


def eval(score_pred_file_name: FileName,
           cpid_resolute_file: FileName,
         n_way=3,
         ):
    topic = "abortion"
    pred_path: FilePath = pjoin(output_path, score_pred_file_name)
    dpid_resolute: Dict[str, DPID] = load_dpid_resolute(cpid_resolute_file)
    score_d: Dict[DPID, np.ndarray] = get_datapoint_score(pred_path, dpid_resolute, "avg")


    def argmax(arr : np.ndarray) -> int:
        return arr.argmax()

    pred_d: Dict[DPID, int] = dict_value_map(argmax, score_d)

    dev_labels = get_dev_labels(topic)
    if n_way == 2:
        def merge_label(e):
            dpid, label = e
            return dpid, {
                0: 0,
                1: 1,
                2: 1,
            }[label]
        dev_labels = lmap(merge_label, dev_labels)

    def fetch_pred(e : Tuple[DPID, int]):
        dpid, label = e
        pred = pred_d[dpid]
        return pred

    gold_list: List[int] = right(dev_labels)
    pred_list: List[int] = lmap(fetch_pred, dev_labels)
    if n_way == 3:
        all_result = eval_3label(gold_list, pred_list)
    elif n_way == 2:
        all_result = eval_2label(gold_list, pred_list)
    else:
        assert False
    print(all_result)
    f1 = sum([result['f1'] for result in all_result]) / n_way
    print("Avg F1 : ", f1)


def load_dpid_resolute(name: FileName):
    dpid_resolute: Dict[str, DPID] = pickle.load(open(pjoin(output_path, name), "rb"))
    return dpid_resolute
