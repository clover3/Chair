import json
import os
from typing import Iterable, Dict, List

from arg.perspectives.load import splits
from arg.perspectives.qck.qck_common import get_qck_candidate_from_ranked_list
from arg.qck.decl import QCKCandidate
from cpath import data_path
from list_lib import lmap
from misc_lib import merge_dict_list
from trec.trec_parse import load_ranked_list_grouped

split_name1 = "new_split.json"
split_name2 = "new_split2.json"


def load_split_info_json(split_json_filename):
    split_info_path = os.path.join(data_path, "perspective", split_json_filename)
    return json.load(open(split_info_path, "r"))


def get_qids_for_split(split_json_filename, split) -> Iterable[str]:
    info = load_split_info_json(split_json_filename)
    for qid in info:
        if info[qid] == split:
            yield qid


def get_split_size(split_json_filename, split):
    return len(list(get_qids_for_split(split_json_filename, split)))


def get_all_ranked_list() -> Dict[str, List[str]]:
    def get_qres_path(split):
        q_res_path = os.path.join("output",
                                  "perspective_experiments",
                                  "pc_qres",
                                  "{}.txt".format(split))
        return q_res_path

    path_list = lmap(get_qres_path, splits)
    rlg_list = lmap(load_ranked_list_grouped, path_list)
    return merge_dict_list(rlg_list)


def get_qck_candidate_for_split(split_filename, split) -> Dict[str, List[QCKCandidate]]:
    qck_candiate_d: Dict[str, List[QCKCandidate]] = get_qck_candidate_from_ranked_list(get_all_ranked_list())
    d = {}
    for qid in get_qids_for_split(split_filename, split):
        d[qid] = qck_candiate_d[qid]
    return d