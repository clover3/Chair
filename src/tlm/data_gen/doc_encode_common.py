import math
import random
from typing import List, Iterable, Any

def split_by_window(tokens, window_size):
    cursor = 0
    while cursor < len(tokens):
        ed = cursor + window_size
        yield tokens[cursor: ed]
        cursor = ed


def join_tokens(query_tokens, second_tokens):
    out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
    return out_tokens, segment_ids


def seg_selection_by_geo_sampling(g_factor=0.5):
    def seg_selection_fn(segs):
        for idx, seg in enumerate(segs):
            chance = math.pow(g_factor, idx)
            include = idx == 0 or random.random() < chance
            if include:
                yield seg
    return seg_selection_fn


def seg_selection_take_first():
    def seg_selection_fn(segs):
        for idx, seg in enumerate(segs):
            if idx == 0:
                yield seg
            break
    return seg_selection_fn


def draw_window_size(window_size):
    if random.random() < 0.5:
        return int(window_size * max(0.3, random.random()))
    else:
        return window_size


def enum_passage_random_short(tokens: List[Any], window_size: int) -> Iterable[List[Any]]:
    cursor = 0
    while cursor < len(tokens):
        st = cursor
        current_window_size = draw_window_size(window_size)
        ed = cursor + current_window_size

        second_tokens = tokens[st:ed]
        cursor += current_window_size
        yield second_tokens


def split_window_get_length(d_tokens_ids, window_size):
    cursor = 0
    d_seg_len_list = []
    while cursor < len(d_tokens_ids):
        remain = len(d_tokens_ids) - cursor
        l = min(remain, window_size)
        d_seg_len_list.append(l)
        cursor += window_size
    return d_seg_len_list