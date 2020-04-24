import os
from typing import Dict, List, Tuple

from cpath import data_path
from galagos.parse import load_qrels, write_qrels
from list_lib import dict_value_map


def load_clef_qrels() -> Dict[str, List[str]]:
    path1 = os.path.join(data_path, "CLEFeHealth2017IRtask", "assessments", "2017", "clef2017_qrels.txt")
    q_rel_d1 = load_qrels(path1)
    path2 = os.path.join(data_path, "CLEFeHealth2017IRtask", "assessments", "2016", "task1.qrels")
    q_rel_d2 = load_qrels(path2)

    def fn(pair_list):
        return list([doc_id for doc_id, score in pair_list if score > 0])
    q_rel_1 = dict_value_map(fn, q_rel_d1)
    q_rel_2 = dict_value_map(fn, q_rel_d2)

    for key in q_rel_2:
        q_rel_1[key].extend(q_rel_2[key])

    return q_rel_1

def combine_qrels():
    path1 = os.path.join(data_path, "CLEFeHealth2017IRtask", "assessments", "2017", "clef2017_qrels.txt")
    q_rel_d1 = load_qrels(path1)
    path2 = os.path.join(data_path, "CLEFeHealth2017IRtask", "assessments", "2016", "task1.qrels")
    q_rel_d2 = load_qrels(path2)

    combined: Dict[str, List[Tuple[str, int]]] = {}
    for key in q_rel_d2:
        concat_list = q_rel_d2[key] + q_rel_d1[key]
        new_list = {}
        for doc_id, score in concat_list:
            new_list[doc_id] = int(score)

        l: List[Tuple[str, int]] = list(new_list.items())
        combined[key] = l


    save_path = os.path.join(data_path, "CLEFeHealth2017IRtask", "combined.qrels")
    write_qrels(combined, save_path)
