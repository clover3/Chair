import json
import os
from typing import List
from typing import NamedTuple

from cpath import output_path


def load_alamri1_problem_info_json():
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "biobert_alamri1.info")
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    return info_d


class AlamriProblem(NamedTuple):
    data_id: int
    group_no: int
    inner_idx: int
    text1: str
    text2: str


def load_alamri_problem() -> List[AlamriProblem]:
    info_d = load_alamri1_problem_info_json()
    output = []
    for data_id_s, info in info_d.items():
        p = AlamriProblem(data_id_s, info['group_no'], info['inner_idx'],
                      info['text1'], info['text2'])
        output.append(p)

    assert len(output) == 305
    return output