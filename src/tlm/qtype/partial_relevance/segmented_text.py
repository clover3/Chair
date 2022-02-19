from typing import NamedTuple, List, Iterator, Tuple

import numpy as np

from cache import named_tuple_to_json
from data_generator.tokenizer_wo_tf import ids_to_text, pretty_tokens


class SegmentedText(NamedTuple):
    tokens_ids: List[int]
    seg_token_indices: List[List[int]]

    def get_seg_len(self):
        return len(self.seg_token_indices)

    def enum_seg_idx(self):
        yield from range(self.get_seg_len())

    def get_empty_seg_mask(self):
        return np.zeros([self.get_seg_len()], np.int)

    def enum_token_idx_from_seg_idx(self, seg_idx) -> Iterator[int]:
        yield from self.seg_token_indices[seg_idx]

    def get_token_idx_as_head_tail(self, seg_idx) -> Tuple[List[int], List[int]]:
        indice = self.seg_token_indices[seg_idx]
        prev_idx = None
        split_idx = len(indice)
        for j, idx in enumerate(indice):
            if prev_idx is not None:
                if prev_idx == idx-1:
                    pass
                else:
                    split_idx = j

        return indice[:split_idx], indice[split_idx:]

    def get_tokens_for_seg(self, seg_idx):
        return [self.tokens_ids[i] for i in self.seg_token_indices[seg_idx]]

    def get_dropped_text(self, drop_indices):
        new_seg = []
        new_seg_indices = []
        offset = 0
        for seg_idx in range(self.get_seg_len()):
            if seg_idx in drop_indices:
                offset = offset - len(self.seg_token_indices[seg_idx])
            else:
                for i in self.enum_token_idx_from_seg_idx(seg_idx):
                    new_seg.append(self.tokens_ids[i])

                new_indices = [idx + offset for idx in self.enum_token_idx_from_seg_idx(seg_idx)]
                new_seg_indices.append(new_indices)
        return SegmentedText(new_seg, new_seg_indices)

    def to_json(self):
        return named_tuple_to_json(self)

    @classmethod
    def from_json(cls, j):
        return SegmentedText(j['tokens_ids'], j['seg_token_indices'])

    @classmethod
    def from_tokens_ids(cls, tokens_ids):
        seg_token_indices = [[i] for i in range(len(tokens_ids))]
        return SegmentedText(tokens_ids, seg_token_indices)


    def get_readable_rep(self, tokenizer):
        s_list = []
        for i in self.enum_seg_idx():
            s_out = self.get_segment_tokens_rep(tokenizer, i)
            s_list.append(s_out)
        return " ".join(s_list)

    def get_segment_tokens_rep(self, tokenizer, i):
        s = ids_to_text(tokenizer, self.get_tokens_for_seg(i))
        s_out = f"{i}) {s}"
        return s_out


def get_replaced_segment(s: SegmentedText, drop_indices: List[int], tokens_tbi: List[int]) -> SegmentedText:
    # for i, cur_i in enumerate(drop_indices):
    #     if i > 0:
    #         prev_i = drop_indices[i-1]
    #         if cur_i != prev_i + 1:
    #             print("drop indices are expected to be continuous")
    #             print("All indices will be dropped but new words would be added once")

    new_tokens_added = False
    new_tokens: List[int] = []
    new_indices = []

    def append_tokens(tokens):
        st = len(new_tokens)
        ed = st + len(tokens)
        new_tokens.extend(tokens)
        new_indices.append(list(range(st, ed)))

    for i_ in s.enum_seg_idx():
        assert type(i_) == int
        i: int = i_
        if i not in drop_indices:
            tokens = s.get_tokens_for_seg(i)
            append_tokens(tokens)
        else:
            if new_tokens_added:
                pass
            else:
                new_tokens_added = True
                append_tokens(tokens_tbi)
    return SegmentedText(new_tokens, new_indices)


def print_segment_text(tokenizer, text: SegmentedText):
    for i in range(text.get_seg_len()):
        tokens = text.get_tokens_for_seg(i)
        s = ids_to_text(tokenizer, tokens)
        print("{}:\t{}".format(i, s))


def seg_to_text(tokenizer, segment: SegmentedText) -> str:
    ids: List[int] = segment.tokens_ids
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return pretty_tokens(tokens, True)


def get_highlighted_text(tokenizer, drop_indices, text2: SegmentedText):
    all_words = [ids_to_text(tokenizer, text2.get_tokens_for_seg(i)) for i in range(text2.get_seg_len())]
    for i in drop_indices:
        all_words[i] = "[{}]".format(all_words[i])
    text = " ".join(all_words)
    return text