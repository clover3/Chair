from collections import defaultdict
from typing import Iterator, Tuple, List

from misc_lib import average


def enum_subseq(tokens_length: int, window_size, offset=0) -> Iterator[Tuple[int, int]]:
    st = offset
    while st < tokens_length:
        ed = min(st + window_size, tokens_length)
        yield st, ed
        st += window_size


def enum_subseq_1(tokens_length: int) -> Iterator[Tuple[int, int]]:
        yield from enum_subseq(tokens_length, 1, 0)


def enum_subseq_136(tokens_length: int) -> Iterator[Tuple[int, int]]:
    for offset in [0, 1, 2]:
        for window_size in [1, 3, 6]:
            yield from enum_subseq(tokens_length, window_size, offset)


def enum_subseq_ex(tokens_length: int, ex_mask, window_size, offset=0) -> Iterator[Tuple[int, int]]:
    st = offset
    while st < tokens_length:
        ed = min(st + window_size, tokens_length)
        include = True
        for i in range(st, ed):
            if ex_mask[i]:
                include = False
                break
        if include:
            yield st, ed
        st += window_size


def enum_subseq_136_ex(tokens_length: int, ex_mask) -> Iterator[Tuple[int, int]]:
    for offset in [0, 1, 2]:
        for window_size in [1, 3, 6]:
            yield from enum_subseq_ex(tokens_length, ex_mask, window_size, offset)


def token_level_attribution(scores: List[float], intervals: List[Tuple[int, int]]) -> List[float]:
    scores_building = defaultdict(list)

    for s, (st, ed) in zip(scores, intervals):
        for i in range(st, ed):
            scores_building[i].append(s)

    n_seq = max(scores_building.keys()) + 1
    scores = []
    for i in range(n_seq):
        scores.append(average(scores_building[i]))
    return scores