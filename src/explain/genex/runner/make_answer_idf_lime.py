import os
from typing import List, Tuple

import numpy as np

from cache import load_from_pickle
from cpath import output_path
from explain.genex.load import load_as_lines
from misc_lib import get_second


def make_answer(problem: str, scores: List[Tuple[str, float]]) -> List[str]:
    scores.sort(key=get_second, reverse=True)

    max_score = None
    answer = []
    for token, score in scores:
        if token == "[SEP]":
            continue
        if max_score is None:
            max_score = score

        if score < max_score * 0.1:
            break
        answer.append(token)

    # among tokens from documents
    # select unique words that has highest score
    return answer


def save_score_to_file(data, save_path, scores):
    out_f = open(save_path, 'w')
    for problem, score in zip(data, scores):
        answer_tokens: List[str] = make_answer(problem, score)
        answer = " ".join(answer_tokens)
        out_f.write(answer + "\n")
    out_f.close()


def main():
    for data_name in ["clue", "tdlt"]:
        method = "idflime"
        score_name = "{}_{}".format(data_name, method)
        try:
            save_name = "{}.txt".format(score_name)
            save_path = os.path.join(output_path, "genex", save_name)
            scores: List[np.array] = load_from_pickle(score_name)
            data: List[str] = load_as_lines(data_name)
            save_score_to_file(data, save_path, scores)
        except:
            print(data_name)
            raise


if __name__ == "__main__":
    main()
