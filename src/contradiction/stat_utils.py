from typing import List


def read_line_scores(scores_path) -> List[int]:
    scores = []
    for line in open(scores_path, "r"):
        scores.append(int(line))
    return scores


def compute_correct(labels, preds):
    correct_list = []
    for l, p in zip(labels, preds):
        if l == p :
            correct_list.append(1)
        else:
            correct_list.append(0)
    return correct_list