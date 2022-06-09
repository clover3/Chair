import json
import os
from typing import List, Iterator
from typing import NamedTuple

from cpath import output_path


def load_alamri1_problem_info_json():
    info_path = os.path.join(output_path, "alamri_annotation1", "alamri1.info")
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    return info_d


class AlamriProblem(NamedTuple):
    data_id: int
    group_no: int
    inner_idx: int
    text1: str
    text2: str

    def get_problem_id(self):
        return "{}_{}".format(self.group_no, self.inner_idx)


def load_alamri_problem() -> List[AlamriProblem]:
    info_d = load_alamri1_problem_info_json()
    output = []
    for data_id_s, info in info_d.items():
        p = AlamriProblem(data_id_s, info['group_no'], info['inner_idx'],
                      info['text1'], info['text2'])
        output.append(p)

    assert len(output) == 305
    return output


def get_valid_split_groups() -> List[int]:
    return list(range(13))


def load_alamri_split(split) -> List[AlamriProblem]:
    all_problems = load_alamri_problem()
    valid_groups = get_valid_split_groups()
    if split == "val" or split == "dev":
        return [p for p in all_problems if p.group_no in valid_groups]
    elif split == "test":
        return [p for p in all_problems if p.group_no not in valid_groups]
    else:
        raise ValueError(split)


def iter_unique_text(split) -> Iterator[str]:
    problems: List[AlamriProblem] = load_alamri_split(split)
    seen = set()
    for p in problems:
        for text in [p.text1, p.text2]:
            if text not in seen:
                yield text
                seen.add(text)
            else:
                pass
