import os
from typing import List

from cpath import data_path
from dataset_specific.ists.parse import iSTSProblem, parse_label_file, AlignmentPredictionList, iSTSProblemWChunk


def get_ists_texts_path(genre, split, sent_name):
    assert genre in ["headlines", "images", "answers-students"]
    assert sent_name in ["sent1", "sent2"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testinput.{genre}.{sent_name}.txt")
    else:
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.{sent_name}.txt")
    return p


def get_ists_chunk_path(genre, split, sent_name):
    assert genre in ["headlines", "images", "answers-students"]
    assert sent_name in ["sent1", "sent2"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testinput.{genre}.{sent_name}.chunk.txt")
    else:
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.{sent_name}.chunk.txt")
    return p


def get_ists_label_path(genre, split):
    assert genre in ["headlines", "images", "answers-students"]
    if split == "test":
        p = os.path.join(data_path, "sts_16", "test", f"STSint.testinput.{genre}.wa")
    elif split == "train":
        p = os.path.join(data_path, "sts_16", f"STSint.input.{genre}.wa")
    else:
        raise Exception()
    return p


def load_ists_problems(genre, split) -> List[iSTSProblem]:
    def read_for_part(sent_name):
        lines = open(get_ists_texts_path(genre, split, sent_name), "r").readlines()
        return [l.strip() for l in lines if l.strip()]

    sent1_list: List[str] = read_for_part("sent1")
    sent2_list: List[str] = read_for_part("sent2")
    assert len(sent1_list) == len(sent2_list)

    p_list = []
    for i in range(len(sent1_list)):
        problem_id = str(i+1)
        p = iSTSProblem(problem_id, sent1_list[i], sent2_list[i])
        p_list.append(p)
    return p_list


def load_ists_problems_w_chunk(genre, split) -> List[iSTSProblemWChunk]:
    def read_for_part(sent_name):
        lines = open(get_ists_chunk_path(genre, split, sent_name), "r").readlines()

        def parse_line(line) -> List[str]:
            line = line.strip()
            chunks = line.split("] [")
            assert chunks[0][0] == "["
            chunks[0] = chunks[0][1:]
            assert chunks[-1][-1] == "]"
            chunks[-1] = chunks[-1][:-1]
            for t in chunks:
                assert "[" not in chunks
                assert "]" not in chunks

            chunks = [t.strip() for t in chunks]
            return chunks

        return list(map(parse_line, lines))

    sent1_list: List[List[str]] = read_for_part("sent1")
    sent2_list: List[List[str]] = read_for_part("sent2")
    assert len(sent1_list) == len(sent2_list)
    problems = load_ists_problems(genre, split)
    p_list = []
    n_problem = len(sent1_list)
    for i in range(n_problem):
        p: iSTSProblem = problems[i]
        chunks1 = sent1_list[i]
        chunks2 = sent2_list[i]
        assert " ".join(chunks1) == p.text1
        assert " ".join(chunks2) == p.text2

        def get_chunk_ids(chunks) -> List[List[int]]:
            idx = 1
            ids = []
            for chunk in chunks:
                n_tokens = len(chunk.split())
                ids.append([j+idx for j in range(n_tokens)])
                idx = idx + n_tokens
            return ids
        p_new = iSTSProblemWChunk(p.problem_id, p.text1, p.text2,
                                  chunks1, chunks2,
                                  get_chunk_ids(chunks1), get_chunk_ids(chunks2))
        p_list.append(p_new)
    return p_list


def load_ists_label(genre, split) -> AlignmentPredictionList:
    return parse_label_file(get_ists_label_path(genre, split))


