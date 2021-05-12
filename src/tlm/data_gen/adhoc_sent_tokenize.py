from typing import List, Iterable, Tuple, Any

from data_generator.tokenizer_wo_tf import get_tokenizer
from nltk import sent_tokenize

from list_lib import lmap
from tlm.data_gen.doc_encode_common import draw_window_size


def enum_passage(title_tokens: List[Any], body_tokens: List[List[Any]], window_size: int) -> Iterable[List[Any]]:
    cursor = 0
    if len(title_tokens) > window_size * 0.9:
        raise ValueError

    while cursor < len(body_tokens):
        tokens = []
        tokens.extend(title_tokens)

        while len(tokens) + len(body_tokens[cursor]) <= window_size:
            tokens.extend(body_tokens[cursor])
            cursor += 1
        yield tokens



def enum_passage_max_20(title_tokens: List[Any], body_tokens: List[List[Any]], window_size: int) -> Iterable[List[Any]]:
    cursor = 0
    if len(title_tokens) > window_size * 0.9:
        raise ValueError

    while cursor < len(body_tokens):
        tokens = []
        tokens.extend(title_tokens)

        while len(tokens) + len(body_tokens[cursor]) <= window_size:
            tokens.extend(body_tokens[cursor])
            cursor += 1
        yield tokens


def enum_passage_random_short(title_tokens: List[Any], body_tokens: List[List[Any]], window_size: int) -> Iterable[List[Any]]:
    cursor = 0

    while cursor < len(body_tokens):
        tokens = []
        tokens.extend(title_tokens)
        current_window_size = draw_window_size(window_size)
        assert current_window_size > 10
        while cursor < len(body_tokens) and \
                len(tokens) + len(body_tokens[cursor]) <= current_window_size:
            tokens.extend(body_tokens[cursor])
            cursor += 1
        yield tokens


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


class FromTextEncoder:
    def __init__(self, max_seq_length, random_short=False, seg_selection_fn=None, max_seg_per_doc=999999):
        self.max_seq_length = max_seq_length
        self.tokenizer = get_tokenizer()
        self.title_token_max = 64
        self.query_token_max = 64
        if random_short:
            self.enum_passage_fn = enum_passage_random_short
        else:
            self.enum_passage_fn = enum_passage

        self.seg_selection_fn = seg_selection_fn
        self.max_seg_per_doc = max_seg_per_doc

    def encode(self, query_tokens, title, body) -> List[Tuple[List, List]]:
        query_tokens = query_tokens[:self.query_token_max]

        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 0

        title_tokens = self.tokenizer.tokenize(title)
        title_tokens = title_tokens[:self.title_token_max]

        maybe_max_body_len = content_len - len(title_tokens)
        body_tokens: List[List[str]] = self.get_tokens_sent_grouped(body, maybe_max_body_len)
        ##
        if not body_tokens:
            body_tokens: List[List[str]] = [["[PAD]"]]

        n_tokens = sum(map(len, body_tokens))
        insts = []
        for idx, second_tokens in enumerate(self.enum_passage_fn(title_tokens, body_tokens, content_len)):
            out_tokens, segment_ids = join_tokens(query_tokens, second_tokens)
            entry = out_tokens, segment_ids
            insts.append(entry)
            assert idx <= n_tokens
            if idx >= self.max_seg_per_doc:
                break

        if self.seg_selection_fn is not None:
            insts = self.seg_selection_fn(insts)

        return insts

    def get_tokens_sent_grouped(self, body, maybe_max_body_len) -> List[List[str]]:
        body_sents = sent_tokenize(body)
        body_tokens: List[List[str]] = lmap(self.tokenizer.tokenize, body_sents)
        body_tokens: List[List[str]] = list([tokens for tokens in body_tokens if tokens])
        out_body_tokens: List[List[str]] = []
        for tokens in body_tokens:
            if len(tokens) >= maybe_max_body_len:
                out_body_tokens.extend(split_by_window(tokens, maybe_max_body_len))
            else:
                out_body_tokens.append(tokens)

        return out_body_tokens


class SeroFromTextEncoder:
    def __init__(self, src_window_size, max_seq_length, random_short=True, max_seg_per_doc=4):
        def seg_selection_fn(insts: List[Tuple[List, List]]) -> List[Tuple[List, List]]:
            all_tokens = []
            all_seg_ids = []
            for idx, (tokens, seg_ids) in enumerate(insts):
                all_tokens.extend(tokens)
                all_seg_ids.extend(seg_ids)
                if idx == max_seg_per_doc:
                    break
            return [(all_tokens[:max_seq_length], all_seg_ids[:max_seq_length])]

        assert src_window_size <= max_seq_length
        self.inner_encoder = FromTextEncoder(src_window_size, random_short, seg_selection_fn, max_seg_per_doc)

    def encode(self, query_tokens, title, body) -> List[Tuple[List, List]]:
        return self.inner_encoder.encode(query_tokens, title, body)
