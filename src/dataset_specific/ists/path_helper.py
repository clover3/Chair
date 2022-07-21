import os
from typing import List

from cpath import data_path
from dataset_specific.ists.parse import ISTSProblem, parse_label_file, AlignmentList


def get_ists_texts_path(genre, split, sent_name):
    assert genre in ["headlines", "images", "answers-students"]
    assert sent_name in ["sent1", "sent2"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testInput.{genre}.{sent_name}.txt")
    else:
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.{sent_name}.txt")
    return p


def get_ists_label_path(genre, split):
    assert genre in ["headlines", "images", "answers-students"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testInput.{genre}.wa")
    else:
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.wa")
    return p


def load_ists_problems(genre, split) -> List[ISTSProblem]:
    def read_for_part(sent_name):
        lines = open(get_ists_texts_path(genre, split, sent_name), "r").readlines()
        return [l.strip() for l in lines if l.strip()]

    sent1_list: List[str] = read_for_part("sent1")
    sent2_list: List[str] = read_for_part("sent2")
    assert len(sent1_list) == len(sent2_list)

    p_list = []
    for i in range(len(sent1_list)):
        problem_id = str(i+1)
        p = ISTSProblem(problem_id, sent1_list[i], sent2_list[i])
        p_list.append(p)
    return p_list


def load_ists_label(genre, split) -> AlignmentList:
    return parse_label_file(get_ists_label_path(genre, split))


