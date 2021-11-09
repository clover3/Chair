import abc
from typing import List, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from tlm.data_gen.adhoc_sent_tokenize import enum_passage_random_short, enum_passage


class QueryDocumentEncoderI:
    @abc.abstractmethod
    def encode(self, query_tokens: List[str], title_tokens: List[str], body_tokens_list: List[List[str]]) -> List[Tuple[List, List]]:
        pass


class QueryDocumentEncoder(QueryDocumentEncoderI):
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

    def encode(self, query_tokens: List[str], title_tokens: List[str], body_tokens_list: List[List[str]]) -> List[Tuple[List, List]]:
        query_tokens = query_tokens[:self.query_token_max]
        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 0

        title_tokens = title_tokens[:self.title_token_max]
        if not body_tokens_list:
            body_tokens_list: List[List[str]] = [["[PAD]"]]

        n_tokens = sum(map(len, body_tokens_list))
        insts = []
        for idx, second_tokens in enumerate(self.enum_passage_fn(title_tokens, body_tokens_list, content_len)):
            entry = query_tokens, second_tokens
            insts.append(entry)
            if idx > n_tokens:
                print('idx', idx)
                print(lmap(len, body_tokens_list))
            assert idx <= n_tokens
            if idx >= self.max_seg_per_doc:
                break

        if self.seg_selection_fn is not None:
            insts = self.seg_selection_fn(insts)

        return insts

