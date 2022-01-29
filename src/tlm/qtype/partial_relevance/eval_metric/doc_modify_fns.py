from typing import Callable, List

import numpy as np

from tlm.qtype.partial_relevance.eval_data_structure import SegmentedText, get_replaced_segment

DocModFunc = Callable[[SegmentedText, List[float]], SegmentedText]
DocReplaceFunc = Callable[[SegmentedText, List[float], List[int]], SegmentedText]
QueryReplaceFunc = Callable[[SegmentedText, int, List[int]], SegmentedText]


def get_top_k_fn(k) -> DocModFunc:
    def get_top_k_drop_inner(text: SegmentedText, scores: List[float]) -> SegmentedText:
        seg_len = text.get_seg_len()
        drop_len = int(seg_len * k)
        # print("segment scores:", lmap(two_digit_float, scores))
        #
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))
        argsorted = np.argsort(scores)
        drop_indices = argsorted[-drop_len:]  # Pick one with highest scores
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return get_top_k_drop_inner


def get_drop_non_zero() -> DocModFunc:
    def drop_non_zero(text: SegmentedText, scores: List[float]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        drop_indices = [idx for idx, s in enumerate(scores) if s > 1e-8]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_non_zero


def get_drop_zero() -> DocModFunc:
    def drop_zero(text: SegmentedText, scores: List[float]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        drop_indices = [idx for idx, s in enumerate(scores) if abs(s) < 1e-8]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_zero


def assert_float_or_int(v):
    assert type(v) == float or type(v) == int


def get_replace_non_zero() -> DocReplaceFunc:
    def replace_non_zero(text: SegmentedText, scores: List[float], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        assert type(word[0]) == int

        drop_indices = [idx for idx, s in enumerate(scores) if s > 1e-8]
        return get_replaced_segment(text, drop_indices, word)
    return replace_non_zero


def get_replace_zero() -> DocReplaceFunc:
    def replace_zero(text: SegmentedText, scores: List[float], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        assert type(word[0]) == int

        drop_indices = [idx for idx, s in enumerate(scores) if abs(s) < 1e-8]
        return get_replaced_segment(text, drop_indices, word)
    return replace_zero

