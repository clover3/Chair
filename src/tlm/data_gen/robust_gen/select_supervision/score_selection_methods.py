import random
from typing import List

import numpy as np

from misc_lib import pick1


def get_target_indices_get_best(passage_scores: List) -> List[int]:
    selected_idx = int(np.argmax(passage_scores))
    return [selected_idx]


def get_target_indices_all(passage_scores) -> List[int]:
    return [idx for idx, _ in enumerate(passage_scores)]


def get_target_indices_first_and_best(passage_scores) -> List[int]:
    best_idx = int(np.argmax(passage_scores))

    if best_idx == 0:
        return [0]
    else:
        return [0, best_idx]


def get_target_indices_random_over_09(passage_scores) -> List[int]:
    candidate = []
    for idx, s in enumerate(passage_scores):
        if s > 0.9:
            candidate.append(idx)

    if random.random() < 0.1:
        output = [0]
        if candidate:
            output.append(pick1(candidate))
        return output
    else:
        return [0]


def get_target_indices_best_or_over_09(passage_scores) -> List[int]:
    best_idx = int(np.argmax(passage_scores))
    indices = []
    for idx, s in enumerate(passage_scores):
        if s > 0.9:
            indices.append(idx)

    if not indices:
        indices.append(best_idx)
    return indices

