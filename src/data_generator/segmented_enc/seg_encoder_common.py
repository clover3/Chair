from abc import ABC, abstractmethod
from typing import Tuple, List

from arg.qck.encode_common import encode_single
from misc_lib import CountWarning


class EncoderInterface(ABC):
    @abstractmethod
    def encode(self, tokens) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        pass

    @abstractmethod
    def encode_from_text(self, text) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        pass


class SingleEncoder(EncoderInterface):
    def __init__(self, tokenizer, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()

    def encode(self, tokens) -> Tuple[List, List, List]:
        if len(tokens) > self.max_seq_length - 2:
            self.counter_warning.add_warn()
        return encode_single(self.tokenizer, tokens, self.max_seq_length)

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


class EvenSplitEncoder(EncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        middle = int(len(tokens) / 2)
        first = tokens[:middle]
        second = tokens[:middle]
        all_input_ids: List[int] = []
        all_input_mask: List[int] = []
        all_segment_ids: List[int] = []
        if len(tokens) > self.segment_len * 2:
            self.counter_warning.add_warn()

        for sub_tokens in [first, second]:
            input_ids, input_mask, segment_ids = encode_single(self.tokenizer, sub_tokens, self.segment_len)
            all_input_ids.extend(input_ids)
            all_input_mask.extend(input_mask)
            all_segment_ids.extend(segment_ids)

        return all_input_ids, all_input_mask, all_segment_ids

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))