import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Set

from arg.qck.encode_common import encode_single
from data_generator.tokenizer_wo_tf import get_continuation_voca_ids, FullTokenizer
from data_generator2.segmented_enc.sent_split_by_spacy import split_spacy_tokens
from misc_lib import CountWarning, SuccessCounter
from tlm.data_gen.base import get_basic_input_feature_as_list, combine_with_sep_cls, concat_triplet_windows, \
    combine_with_sep_cls2


def is_only_0_or_1(l):
    for v in l:
        if v != 0 and v !=1 :
            return False
    return True


class SingleEncoderInterface(ABC):
    # @abstractmethod
    # def encode(self, tokens) -> Tuple[List, List, List]:
    #     # returns input_ids, input_mask, segment_ids
    #     pass

    @abstractmethod
    def encode_from_text(self, text) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        pass


class PairEncoderInterface(ABC):
    # @abstractmethod
    # def encode(self, tokens) -> Tuple[List, List, List]:
    #     # returns input_ids, input_mask, segment_ids
    #     pass

    @abstractmethod
    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        pass


class SingleEncoder(SingleEncoderInterface):
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


class BasicConcatEncoder(PairEncoderInterface):
    def __init__(self, tokenizer, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()

    def encode(self, tokens1, tokens2) -> Tuple[List, List, List]:
        tokens, segment_ids = combine_with_sep_cls(self.max_seq_length, tokens1, tokens2)
        triplet = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_length, tokens, segment_ids)
        return triplet

    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))


class BasicConcatEncoder2(PairEncoderInterface):
    def __init__(self, tokenizer, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()

    def encode(self, tokens1, tokens2) -> Tuple[List, List, List]:
        tokens, segment_ids = combine_with_sep_cls2(self.max_seq_length, tokens1, tokens2)
        triplet = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_length, tokens, segment_ids)
        return triplet

    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))


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


class EvenSplitEncoder(SingleEncoderInterface):
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
        second = tokens[middle:]
        if len(tokens) > self.segment_len * 2:
            self.counter_warning.add_warn()

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


def split_even_avoid_cut(tokens):
    cut_loc = int(len(tokens) / 2)
    if cut_loc == 0:
        pass
    else:
        while tokens[cut_loc - 1].endswith("##"):
            cut_loc += 1
            assert cut_loc <= len(tokens)
    return cut_loc


class EvenSplitEncoderAvoidCut(SingleEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        cut_loc = split_even_avoid_cut(tokens)

        first = tokens[:cut_loc]
        second = tokens[cut_loc:]
        if len(tokens) > self.segment_len * 2:
            self.counter_warning.add_warn()

        return encode_two_segments(self.tokenizer, self.segment_len,
                                   first, second)

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


class SplitBySegmentIDs(SingleEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        cut_loc = split_even_avoid_cut(tokens)

        first = tokens[:cut_loc]
        second = tokens[cut_loc:]
        if len(tokens) > self.segment_len * 2:
            self.counter_warning.add_warn()

        return encode_two_no_sep(self.tokenizer, self.segment_len, first, second)

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


class SpacySplitEncoder(SingleEncoderInterface):
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


class SpacySplitEncoder2(SingleEncoderInterface):
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


class SpacySplitEncoderSlash(SingleEncoderInterface):
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



class SpacySplitEncoderMaskReplacer(SingleEncoderInterface):
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


def encode_two_no_sep(tokenizer, segment_len, tokens1, tokens2):
    max_seq_length = segment_len * 2
    effective_length = max_seq_length - 2   # 100-2 = 98
    effective_length_half = int(effective_length / 2) # 98/2 = 49
    tokens1 = tokens1[:effective_length_half]   # <=49
    tokens2 = tokens2[:effective_length_half]   # <=49
    tokens = tokens1 + tokens2 + ["[SEP]"]  # 49 + 49 + 1 = 99
    segment_ids = [0] * len(tokens1) + [1] * (len(tokens2) + 1)
    assert len(tokens) == len(segment_ids)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens, segment_ids)

    return input_ids, input_mask, segment_ids


def get_random_split_location(tokens) -> Tuple[int, int]:
    retry = True
    n_retry = 0
    while retry:
        st = random.randint(0, len(tokens) - 1)
        while st-1 >= 0 and tokens[st - 1].endswith("##"):
            st += 1
            assert st <= len(tokens)

        # st is located at end of the text
        if st + 1 > len(tokens) and n_retry < 4:
            n_retry += 1
            retry = True
            continue

        ed = random.randint(st+1, len(tokens))
        retry = False
        return st, ed


def random_token_split(tokens):
    st, ed = get_random_split_location(tokens)
    first_a = tokens[:st]
    first_b = tokens[ed:]
    first = first_a + ["[MASK]"] + first_b
    second = tokens[st:ed]
    return first, second


class UnEvenSlice(SingleEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning("One segment is empty")
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        first, second = random_token_split(tokens)
        if not second:
            self.counter_warning.add_warn()
        return encode_two_segments(self.tokenizer, self.segment_len, first, second)

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


class TwoSegConcatEncoder(PairEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len

        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens1, tokens2) -> Tuple[List, List, List]:
        tokens2_first, tokens2_second = random_token_split(tokens2)
        return self.two_seg_concat_core(tokens1, tokens2_first, tokens2_second)

    def two_seg_concat_core(self, tokens1, tokens2_first, tokens2_second) -> Tuple[List, List, List]:
        triplet_list = []
        for part_of_tokens2 in [tokens2_first, tokens2_second]:
            tokens, segment_ids = combine_with_sep_cls(self.segment_len, tokens1, part_of_tokens2)
            if len(tokens) > self.segment_len:
                self.counter_warning.add_warn()

            triplet = get_basic_input_feature_as_list(self.tokenizer, self.segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        return concat_triplet_windows(triplet_list, self.segment_len)

    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))


class TwoSegConcatEncoderQD(PairEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len

        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, d_tokens, q_tokens) -> Tuple[List, List, List]:
        q_tokens_first, q_tokens_second = random_token_split(q_tokens)
        return self.two_seg_concat_core(d_tokens, q_tokens_first, q_tokens_second)

    def two_seg_concat_core(self, d_tokens, q_tokens_first, q_tokens_second) -> Tuple[List, List, List]:
        triplet_list = []
        for part_of_q_tokens in [q_tokens_first, q_tokens_second]:
            tokens, segment_ids = combine_with_sep_cls(self.segment_len, part_of_q_tokens, d_tokens)
            if len(tokens) > self.segment_len:
                self.counter_warning.add_warn()

            triplet = get_basic_input_feature_as_list(self.tokenizer, self.segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        return concat_triplet_windows(triplet_list, self.segment_len)

    def encode_from_text(self, d_text, q_text) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(d_text), self.tokenizer.tokenize(q_text))


class TwoSegConcatRoleEncoder(PairEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len

        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens1, tokens2) -> Tuple[List, List, List]:
        st, ed = get_random_split_location(tokens2)
        first_a = tokens2[:st]
        first_b = tokens2[ed:]
        first = first_a + ["[MASK]"] + first_b
        second = tokens2[st:ed]

        tokens2_first, tokens2_second = random_token_split(tokens2)
        return self.two_seg_concat_core(tokens1, tokens2_first, tokens2_second)

    def two_seg_concat_core(self, tokens1, tokens2_first, tokens2_second):
        triplet_list = []
        for part_of_tokens2 in [tokens2_first, tokens2_second]:
            tokens, segment_ids = combine_with_sep_cls(self.segment_len, tokens1, part_of_tokens2)
            if len(tokens) > self.segment_len:
                self.counter_warning.add_warn()

            triplet = get_basic_input_feature_as_list(self.tokenizer, self.segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        return concat_triplet_windows(triplet_list, self.segment_len)

    def encode_from_text(self, text1, text2):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))


class SingleChunkIndicatingEncoder(SingleEncoderInterface):
    def __init__(self, tokenizer, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer: FullTokenizer = tokenizer
        self.counter_warning = CountWarning()
        self.continue_voca: Set[int] = set(get_continuation_voca_ids(tokenizer))
        self.chunk_start_rate = SuccessCounter()

    def encode(self, tokens) -> Tuple[List, List, List]:
        if len(tokens) > self.max_seq_length - 2:
            self.counter_warning.add_warn()
        input_ids, input_mask, _ = encode_single(self.tokenizer, tokens, self.max_seq_length)

        def is_chunk_start(token_id):
            if token_id in self.continue_voca:
                if token_id != 0:
                    self.chunk_start_rate.fail()
                return False
            else:
                if token_id != 0:
                    self.chunk_start_rate.suc()
                return True

        chunk_start_indicators = list(map(is_chunk_start, input_ids))
        fake_segment_ids = chunk_start_indicators
        return input_ids, input_mask, fake_segment_ids


class ChunkIndicatingEncoder(SingleEncoderInterface):
    def __init__(self, tokenizer, max_seq_length, typical_chunk_len):
        self.max_seq_length = max_seq_length
        self.tokenizer: FullTokenizer = tokenizer
        self.counter_warning = CountWarning()
        self.continue_voca: Set[int] = set(get_continuation_voca_ids(tokenizer))
        self.chunk_start_rate = SuccessCounter()
        self.typical_chunk_len = typical_chunk_len

    def encode(self, tokens) -> Tuple[List, List, List]:
        if len(tokens) > self.max_seq_length - 2:
            self.counter_warning.add_warn()
        input_ids, input_mask, _ = encode_single(self.tokenizer, tokens, self.max_seq_length)

        cur_chunk = []
        chunk_start_indicators = []
        self.chunk_start_rate.suc()
        for idx, token_id in enumerate(input_ids):
            if len(cur_chunk) == 0:
                chunk_start_indicators.append(1)
                self.chunk_start_rate.suc()
            else:
                chunk_start_indicators.append(0)
                self.chunk_start_rate.fail()
            cur_chunk.append(token_id)

            if len(cur_chunk) >= self.typical_chunk_len and token_id not in self.continue_voca:
                cur_chunk = []

        if not is_only_0_or_1(chunk_start_indicators):
            raise ValueError()
        assert len(chunk_start_indicators) == len(input_ids)
        fake_segment_ids = chunk_start_indicators
        return input_ids, input_mask, fake_segment_ids

    def encode_from_text(self, text):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text))


def get_n_random_split_location(tokens, k) -> List[int]:
    good_loc = []
    for i in range(len(tokens)):
        cur_t = tokens[i]
        if not cur_t.endswith("##") and i+1 < len(tokens):
            good_loc.append(i+1)

    locations = random.sample(good_loc, min(k, len(good_loc)))
    return locations


def n_seg_random_split(tokens, n) -> List[List[str]]:
    locations = get_n_random_split_location(tokens, n-1)
    locations.sort()
    cursor = 0
    seg_list = []
    for i, loc in enumerate(locations):
        seg = tokens[cursor: loc]
        assert loc > cursor
        cursor = loc
        if i > 0:
            seg = ["[MASK]"] + seg

        seg = seg + ["[MASK]"]
        seg_list.append(seg)

    seg = tokens[cursor:]
    if locations:
        seg = ["[MASK]"] + seg
    seg_list.append(seg)
    assert len(seg_list) <= n
    return seg_list


class NSegConcatEncoder(PairEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length, n_segment):
        segment_len = int(total_max_seq_length / n_segment)
        self.segment_len = segment_len
        self.n_segment = n_segment

        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens1, tokens2) -> Tuple[List, List, List]:
        seg_list: List[List[str]] = n_seg_random_split(tokens2, self.n_segment)
        while len(seg_list) < self.n_segment:
            empty_seq = []
            seg_list.append(empty_seq)

        assert len(seg_list) == self.n_segment

        triplet_list = []
        for part_of_tokens2 in seg_list:
            tokens, segment_ids = combine_with_sep_cls2(self.segment_len, tokens1, part_of_tokens2)
            if len(tokens) > self.segment_len:
                self.counter_warning.add_warn()

            triplet = get_basic_input_feature_as_list(self.tokenizer, self.segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        return concat_triplet_windows(triplet_list, self.segment_len)

    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))

