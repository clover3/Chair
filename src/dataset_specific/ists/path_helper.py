import os
from typing import List

from cpath import data_path
from dataset_specific.ists.parse import parse_label_file, AlignmentPredictionList, iSTSProblemWChunk, parse_chunks, \
    join_problem_w_chunks, iSTSProblem, load_ists_problem_w_path
from dataset_specific.ists.split_info import ists_genre_list


def get_ists_texts_path(genre, split, sent_name):
    assert genre in ists_genre_list
    assert sent_name in ["sent1", "sent2"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testinput.{genre}.{sent_name}.txt")
    else:
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.{sent_name}.txt")
    return p


def get_ists_chunk_path(genre, split, sent_name):
    assert genre in ists_genre_list
    assert sent_name in ["sent1", "sent2"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testinput.{genre}.{sent_name}.chunk.txt")
    else:
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.{sent_name}.chunk.txt")
    return p


def get_ists_label_path(genre, split):
    assert genre in ists_genre_list
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testinput.{genre}.wa")
    elif split == "train":
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.wa")
    else:
        raise Exception()
    return p


def load_ists_label(genre, split) -> AlignmentPredictionList:
    return parse_label_file(get_ists_label_path(genre, split))


def load_ists_problems_w_chunk(genre, split) -> List[iSTSProblemWChunk]:
    def read_for_part(sent_name):
        file_path = get_ists_chunk_path(genre, split, sent_name)
        return parse_chunks(file_path)

    sent1_list: List[List[str]] = read_for_part("sent1")
    sent2_list: List[List[str]] = read_for_part("sent2")
    assert len(sent1_list) == len(sent2_list)
    problems = load_ists_problems(genre, split)
    p_list = join_problem_w_chunks(problems, sent1_list, sent2_list)
    return p_list


def load_ists_problems(genre, split) -> List[iSTSProblem]:
    p1 = get_ists_texts_path(genre, split, "sent1")
    p2 = get_ists_texts_path(genre, split, "sent2")
    return load_ists_problem_w_path(p1, p2)