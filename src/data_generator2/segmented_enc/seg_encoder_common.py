from abc import ABC, abstractmethod
from typing import Tuple, List

from arg.qck.encode_common import encode_single
from data_generator2.segmented_enc.sent_split_by_spacy import split_spacy_tokens
from misc_lib import CountWarning


class EncoderInterface(ABC):
    # @abstractmethod
    # def encode(self, tokens) -> Tuple[List, List, List]:
    #     # returns input_ids, input_mask, segment_ids
    #     pass

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


def encode_two_segments(tokenizer, segment_len, first, second):
    all_input_ids: List[int] = []
    all_input_mask: List[int] = []
    all_segment_ids: List[int] = []
    for sub_tokens in [first, second]:
        input_ids, input_mask, segment_ids = encode_single(tokenizer, sub_tokens, segment_len)
        all_input_ids.extend(input_ids)
        all_input_mask.extend(input_mask)
        all_segment_ids.extend(segment_ids)
    return all_input_ids, all_input_mask, all_segment_ids


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
        if len(tokens) > self.segment_len * 2:
            self.counter_warning.add_warn()

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


class SpacySplitEncoder(EncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        raise NotImplementedError

    def encode_from_text(self, text):
        spacy_tokens = self.nlp(text)
        seg1, seg2 = split_spacy_tokens(spacy_tokens)

        def text_to_tokens(text):
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.segment_len:
                tokens = tokens[:self.segment_len]
                self.counter_warning.add_warn()
            return tokens

        first = text_to_tokens(seg1)
        second = text_to_tokens(seg2)

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)


class SpacySplitEncoder2(EncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length, cache_split):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        self.cache_split = cache_split
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        raise NotImplementedError

    def encode_from_text(self, text):
        seg1, seg2 = self.cache_split[text]

        def text_to_tokens(text):
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.segment_len:
                tokens = tokens[:self.segment_len]
                self.counter_warning.add_warn()
            return tokens

        first = text_to_tokens(seg1)
        second = text_to_tokens(seg2)

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)


class SpacySplitEncoderSlash(EncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length, cache_split):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        self.cache_split = cache_split
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        raise NotImplementedError

    def encode_from_text(self, text):
        seg1, seg2 = self.cache_split[text]

        def text_to_tokens(text):
            text = text.replace("[MASK]", "/")
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.segment_len:
                tokens = tokens[:self.segment_len]
                self.counter_warning.add_warn()
            return tokens

        first = text_to_tokens(seg1)
        second = text_to_tokens(seg2)

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)



class SpacySplitEncoderMaskReplacer(EncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length, cache_split, mask_replacer):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        self.cache_split = cache_split
        self.mask_replacer = mask_replacer
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        raise NotImplementedError

    def encode_from_text(self, text):
        seg1, seg2 = self.cache_split[text]

        def text_to_tokens(text):
            text = text.replace("[MASK]", self.mask_replacer)
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.segment_len:
                tokens = tokens[:self.segment_len]
                self.counter_warning.add_warn()
            return tokens

        first = text_to_tokens(seg1)
        second = text_to_tokens(seg2)

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)


class SpacySplitEncoderMaskSlash(SpacySplitEncoderMaskReplacer):
    def __init__(self, tokenizer, total_max_seq_length, cache_split):
        super(SpacySplitEncoderMaskSlash, self).__init__(tokenizer, total_max_seq_length, cache_split, "/")


class SpacySplitEncoderNoMask(SpacySplitEncoderMaskReplacer):
    def __init__(self, tokenizer, total_max_seq_length, cache_split):
        super(SpacySplitEncoderNoMask, self).__init__(tokenizer, total_max_seq_length, cache_split, "")
