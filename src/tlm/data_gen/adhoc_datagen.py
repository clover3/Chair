from abc import ABC, abstractmethod
from typing import Tuple

import math

from misc_lib import enum_passage, enum_passage_overlap
from tlm.data_gen.doc_encode_common import enum_passage_random_short
from tlm.data_gen.robust_gen.select_supervision.score_selection_methods import *

Tokens = List[str]
SegmentIDs = List[int]

token_first = "[unused3]"
token_mid = "[unused4]"
token_end = "[unused5]"


class EncoderInterface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def encode(self, query, tokens):
        # returns tokens, segmend_ids
        pass


class TitleRepeatInterface(ABC):
    @abstractmethod
    def encode(self, query, title_tokens, body_tokens):
        # returns tokens, segmend_ids
        pass


class EncoderTokenCounterInterface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def count(self, query, tokens) -> int:
        # returns tokens, segmend_ids
        pass


class EncoderTokenCounter2Interface(ABC):
    def __init__(self, max_seq_length):
        pass

    @abstractmethod
    def count(self, query, tokens) -> List[Tuple[List, List, int]]:
        # returns tokens, segmend_ids
        pass


def get_combined_tokens_segment_ids(query_tokens, second_tokens) -> Tuple[Tokens, SegmentIDs]:
    out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
    return out_tokens, segment_ids


class FirstSegmentAsDoc(EncoderInterface):
    def __init__(self, max_seq_length):
        super(FirstSegmentAsDoc, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens):
        content_len = self.max_seq_length - 3 - len(query_tokens)
        second_tokens = tokens[:content_len]
        out_tokens, segment_ids = get_combined_tokens_segment_ids(query_tokens, second_tokens)
        return [(out_tokens, segment_ids)]


class MultiWindow(EncoderInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindow, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        self.all_segment_as_doc = AllSegmentAsDoc(src_window_size)

    def encode(self, query_tokens, tokens):
        insts = self.all_segment_as_doc.encode(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        for tokens, segment_ids in insts:
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)

            if len(out_tokens) > self.max_seq_length:
                break

        if len(out_segment_ids) == 0:
            print("query:", query_tokens)
            print("doc: ", tokens)
            raise ValueError

        return [(out_tokens[:self.max_seq_length], out_segment_ids[:self.max_seq_length])]


class MultiWindowTokenCount(EncoderTokenCounterInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindowTokenCount, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        self.all_segment_as_doc = AllSegmentAsDocTokenCounter(src_window_size)

    def count(self, query_tokens, tokens):
        insts = self.all_segment_as_doc.count(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        num_content_tokens_acc = 0
        for tokens, segment_ids, num_content_tokens in insts:
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)
            num_content_tokens_acc += num_content_tokens
            if len(out_tokens) > self.max_seq_length:
                break

        return num_content_tokens_acc



class MultiWindowOverlap(EncoderInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindowOverlap, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        step_size = int(self.window_size / 2)
        self.sub_encoder = OverlappingSegments(src_window_size, step_size)

    def encode(self, query_tokens, tokens):
        insts = self.sub_encoder.encode(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        for idx, (tokens, segment_ids) in enumerate(insts):
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)

            if idx + 1 < len(insts):
                assert len(segment_ids) == self.window_size

            if len(out_tokens) > self.max_seq_length:
                break

        return [(out_tokens[:self.max_seq_length], out_segment_ids[:self.max_seq_length])]


class MultiWindowAllSeg(EncoderInterface):
    def __init__(self, src_window_size, max_seq_length):
        super(MultiWindowAllSeg, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.window_size = src_window_size
        self.all_segment_as_doc = AllSegmentAsDoc(src_window_size)

    def encode(self, query_tokens, tokens):
        insts = self.all_segment_as_doc.encode(query_tokens, tokens)
        out_tokens = []
        out_segment_ids = []
        for tokens, segment_ids in insts:
            out_tokens.extend(tokens)
            out_segment_ids.extend(segment_ids)

            if len(out_tokens) == self.max_seq_length:
                yield out_tokens, out_segment_ids
                out_tokens = []
                out_segment_ids = []

            if len(out_tokens) > self.max_seq_length:
                assert False

        if out_tokens:
            yield out_tokens, out_segment_ids


class AllSegmentAsDoc(EncoderInterface):
    def __init__(self, max_seq_length):
        super(AllSegmentAsDoc, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        if len(query_tokens) > 64:
            query_tokens = query_tokens[:64]
        content_len = self.max_seq_length - 3 - len(query_tokens)
        if not tokens:
            tokens = ['[PAD]']
        insts = []
        for second_tokens in enum_passage(tokens, content_len):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class AllSegmentWMark(EncoderInterface):
    def __init__(self, max_seq_length):
        super(AllSegmentWMark, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        if len(query_tokens) > 64:
            query_tokens = query_tokens[:64]
        content_len = self.max_seq_length - 3 - len(query_tokens) - 1
        insts = []
        if not tokens:
            tokens = ['[PAD]']
        passages = list(enum_passage(tokens, content_len))
        for idx, second_tokens in enumerate(passages):
            if idx == 0:
                mark = token_first
            elif idx == len(passages) - 1:
                mark = token_end
            else:
                mark = token_mid
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]", mark] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 2)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class AllSegmentRepeatTitle(TitleRepeatInterface):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.max_title_len = int(max_seq_length * 0.1)
        self.long_title_cnt = 0
        self.total_doc_cnt = 0

    def encode(self, query_tokens, title_tokens, body_tokens) -> List[Tuple[List, List]]:
        if len(title_tokens) > self.max_title_len:
            self.long_title_cnt += 1
        self.total_doc_cnt += 1
        title_tokens = title_tokens[:self.max_title_len]
        content_len = self.max_seq_length - 3 - len(query_tokens) - len(title_tokens)
        assert content_len > 5
        insts = []
        for second_tokens in enum_passage(body_tokens, content_len):
            passage_tokens = title_tokens + second_tokens
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + passage_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(passage_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class AllSegmentAsDocTokenCounter(EncoderTokenCounter2Interface):
    def __init__(self, max_seq_length):
        super(AllSegmentAsDocTokenCounter, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def count(self, query_tokens, tokens) -> List[Tuple[List, List, int]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for second_tokens in enum_passage(tokens, content_len):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids, len(second_tokens)
            insts.append(entry)
        return insts


class OverlappingSegments(EncoderInterface):
    def __init__(self, max_seq_length, step_size):
        super(OverlappingSegments, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.step_size = step_size

    def encode(self, query_tokens, tokens) -> List[Tuple[Tokens, SegmentIDs]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for second_tokens in enum_passage_overlap(tokens, content_len, self.step_size, True):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class OverlappingSegmentsEx:
    def __init__(self, max_seq_length, step_size):
        self.max_seq_length = max_seq_length
        self.step_size = step_size

    def encode(self, query_tokens, tokens) -> List[Tuple[int, int, List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        raw_entries = []
        cursor = self.step_size - content_len
        while cursor < len(tokens):
            st = cursor if cursor >= 0 else 0
            ed = cursor + content_len
            second_tokens = tokens[st:ed]
            cursor += self.step_size
            e = st, ed, second_tokens
            raw_entries.append(e)

        short_window = content_len - self.step_size
        cursor = self.step_size - short_window
        while cursor < len(tokens):
            st = cursor if cursor >= 0 else 0
            ed = cursor + short_window
            second_tokens = tokens[st:ed]
            cursor += self.step_size
            e = st, ed, second_tokens
            raw_entries.append(e)

        for st, ed, second_tokens in raw_entries:
            out_tokens, segment_ids = self.decorate_tokens(query_tokens, second_tokens)
            entry = st, ed, out_tokens, segment_ids
            insts.append(entry)
        return insts

    def decorate_tokens(self, query_tokens, second_tokens):
        out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
        pad_length = self.max_seq_length - len(out_tokens)
        out_tokens += ["[PAD]"] * pad_length
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1) + [0] * pad_length
        assert len(out_tokens) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        return out_tokens, segment_ids


class PassageSampling(EncoderInterface):
    def __init__(self, max_seq_length):
        super(PassageSampling, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == 0:
                include = True
            else:
                include = random.random() < 0.1
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class FirstAndRandom(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(FirstAndRandom, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == 0:
                include = True
            else:
                include = random.random() < 0.1
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class FirstAndRandomTitleRepeat(TitleRepeatInterface):
    def __init__(self, max_seq_length):
        self.all_seg_repeat_encoder = AllSegmentRepeatTitle(max_seq_length)

    def encode(self, query_tokens, title_tokens, body_tokens) -> List[Tuple[List, List]]:
        insts = self.all_seg_repeat_encoder.encode(query_tokens, title_tokens, body_tokens)
        selected_insts = []
        for idx in range(len(insts)):
            if idx == 0:
                include = True
            else:
                include = random.random() < 0.1

            if include:
                selected_insts.append(insts[idx])
        return insts


class FirstNoTitle(TitleRepeatInterface):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.long_title_cnt = 0
        self.total_doc_cnt = 0

    def encode(self, query_tokens, title_tokens, body_tokens) -> List[Tuple[List, List]]:
        self.total_doc_cnt += 1
        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 5
        insts = []
        for second_tokens in enum_passage(body_tokens, content_len):
            passage_tokens = second_tokens
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + passage_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(passage_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
            break
        return insts


class FirstAndRandomNoTitle(TitleRepeatInterface):
    def __init__(self, max_seq_length):
        self.all_seg_repeat_encoder = NoTitle(max_seq_length)

    def encode(self, query_tokens, title_tokens, body_tokens) -> List[Tuple[List, List]]:
        insts = self.all_seg_repeat_encoder.encode(query_tokens, title_tokens, body_tokens)
        selected_insts = []
        for idx in range(len(insts)):
            if idx == 0:
                include = True
            else:
                include = random.random() < 0.1

            if include:
                selected_insts.append(insts[idx])
        return insts


class NoTitle(TitleRepeatInterface):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.long_title_cnt = 0
        self.total_doc_cnt = 0

    def encode(self, query_tokens, title_tokens, body_tokens) -> List[Tuple[List, List]]:
        self.total_doc_cnt += 1
        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 5
        insts = []
        for second_tokens in enum_passage(body_tokens, content_len):
            passage_tokens = second_tokens
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + passage_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(passage_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class AllSegmentNoTitle(TitleRepeatInterface):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.long_title_cnt = 0
        self.total_doc_cnt = 0

    def encode(self, query_tokens, title_tokens, body_tokens) -> List[Tuple[List, List]]:
        self.total_doc_cnt += 1
        content_len = self.max_seq_length - 3 - len(query_tokens)
        assert content_len > 5
        insts = []
        for second_tokens in enum_passage(body_tokens, content_len):
            passage_tokens = second_tokens
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + passage_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(passage_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class GeoSampler(EncoderInterface):
    def __init__(self, max_seq_length, g_factor=0.5):
        super(GeoSampler, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.g_factor = g_factor

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            chance = math.pow(self.g_factor, idx)
            include = random.random() < chance
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class GeoSamplerWSegMark(EncoderInterface):
    def __init__(self, max_seq_length, g_factor=0.5):
        super(GeoSamplerWSegMark, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.g_factor = g_factor

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens) - 1
        insts = []
        passages = list(enum_passage(tokens, content_len))
        for idx, second_tokens in enumerate(passages):
            chance = math.pow(self.g_factor, idx)
            include = random.random() < chance
            if include:
                if idx == 0:
                    mark = token_first
                elif idx == len(passages) - 1:
                    mark = token_end
                else:
                    mark = token_mid
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]", mark] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 2)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class LeadingN(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(LeadingN, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == self.num_segment:
                break

            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class LeadingNWithRandomShort(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(LeadingNWithRandomShort, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage_random_short(tokens, content_len)):
            if idx == self.num_segment:
                break

            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
        return insts


class GeoRandomShortSampler(EncoderInterface):
    def __init__(self, max_seq_length, g_factor=0.5):
        super(GeoRandomShortSampler, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.g_factor = g_factor

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.max_seq_length - 3 - len(query_tokens)
        insts = []
        for idx, second_tokens in enumerate(enum_passage_random_short(tokens, content_len)):
            chance = math.pow(self.g_factor, idx)
            include = random.random() < chance
            if include:
                out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
                segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
                entry = out_tokens, segment_ids
                insts.append(entry)
        return insts


class FirstEquiSero(EncoderInterface):
    def __init__(self, max_seq_length, sero_window_size, num_segment):
        super(FirstEquiSero, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment
        self.sero_window_size = sero_window_size

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_per_window = self.sero_window_size - 3 - len(query_tokens)
        sero_content_length = content_per_window * 4
        content_max_len = self.max_seq_length - 3 - len(query_tokens)
        content_len = min(sero_content_length, content_max_len)
        insts = []
        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)
            entry = out_tokens, segment_ids
            insts.append(entry)
            break
        return insts


class LeadingSegmentsCombined(EncoderInterface):
    def __init__(self, max_seq_length, num_segment):
        super(LeadingSegmentsCombined, self).__init__(max_seq_length)
        self.max_seq_length = max_seq_length
        self.num_segment = num_segment
        self.window_size = int(max_seq_length / num_segment)

    def encode(self, query_tokens, tokens) -> List[Tuple[List, List]]:
        content_len = self.window_size - 3 - len(query_tokens)
        tokens_extending = []
        segment_ids_extending = []

        for idx, second_tokens in enumerate(enum_passage(tokens, content_len)):
            if idx == self.num_segment:
                break
            out_tokens = ["[CLS]"] + query_tokens + ["[SEP]"] + second_tokens + ["[SEP]"]
            segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(second_tokens) + 1)

            assert len(tokens_extending) % self.window_size == 0
            assert len(segment_ids_extending) % self.window_size == 0
            tokens_extending.extend(out_tokens)
            segment_ids_extending.extend(segment_ids)
        return [(tokens_extending, segment_ids_extending)]



