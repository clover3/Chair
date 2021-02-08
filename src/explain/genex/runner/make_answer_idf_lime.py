import os
import sys
from typing import List, Tuple

import numpy as np

from cache import load_from_pickle
from explain.genex.common import get_genex_run_save_dir
from explain.genex.load import load_as_lines
from misc_lib import get_second


def make_answer(scores: List[Tuple[str, float]], threshold_factor) -> List[str]:
    scores.sort(key=get_second, reverse=True)

    max_score = None
    answer = []
    skip_terms = [".", ",", "!", "[SEP]"]
    for token, score in scores:
        if token in skip_terms  :
            continue
        if max_score is None:
            max_score = score

        if score < max_score * threshold_factor:
            break
        answer.append(token)

    # among tokens from documents
    # select unique words that has highest score
    return answer


def save_score_to_file(data, save_path, scores, threshold_factor):
    out_f = open(save_path, 'w')
    for problem, score in zip(data, scores):
        answer_tokens: List[str] = make_answer(score, threshold_factor)
        answer = " ".join(answer_tokens)
        out_f.write(answer + "\n")
    out_f.close()


def main():
    #
    data_name = sys.argv[1]
    method = "idflime"

    threshold_factor = 0.1
    score_name = "{}_{}".format(data_name, method)
    try:
        save_name = "{}.txt".format(score_name)
        if len(sys.argv) > 2:
            save_name = sys.argv[2]

        save_path = os.path.join(get_genex_run_save_dir(), save_name)
        scores: List[np.array] = load_from_pickle(score_name)
        data: List[str] = load_as_lines(data_name)
        save_score_to_file(data, save_path, scores, threshold_factor)
        print("Saved at : ", save_path)
    except:
        print(data_name)
        raise


if __name__ == "__main__":
    main()
