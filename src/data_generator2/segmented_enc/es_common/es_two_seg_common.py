from abc import ABC, abstractmethod
from typing import NamedTuple, List
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.evidence_candidates import EvidencePair, get_st_ed


class PairData(NamedTuple):
    segment1: str
    segment2: str
    label: str
    data_id: str

    def get_label_as_int(self):
        pass


class PartitionedSegment(ABC):
    tokens: List[str]

    @abstractmethod
    def get_first(self):
        pass
    
    @abstractmethod
    def get_second(self):
        pass

    def get(self, part_no):
        fn = [self.get_first, self.get_second][part_no]
        return fn()


class RangePartitionedSegment(PartitionedSegment):
    tokens: List[str]
    st: int
    ed: int
    
    def __init__(self, segment, st, ed):
        self.tokens = segment
        self.st = st
        self.ed = ed

    def get_first(self):
        return self.tokens[:self.st], self.tokens[self.ed:]

    def get_second(self):
        return self.tokens[self.st: self.ed]


class IndicesPartitionedSegment(PartitionedSegment):
    tokens: List[str]
    del_indices1: List[int]
    del_indices2: List[int]

    def __init__(self, tokens, del_indices1, del_indices2):
        self.tokens = tokens
        self.del_indices1 = del_indices1
        self.del_indices2 = del_indices2

    def get_partition_seg(self, segment_idx: int) -> List[str]:
        assert segment_idx == 0 or segment_idx == 1

        tokens_new = list(self.tokens)
        del_indices = [self.del_indices1, self.del_indices2][segment_idx]
        for i in del_indices:
            tokens_new[i] = "[MASK]"
        return tokens_new

    def get_first(self):
        return self.get_partition_seg(0)

    def get_second(self):
        return self.get_partition_seg(1)


class MaskPartitionedSegment(PartitionedSegment):
    part1: List[str]
    part2: List[str]

    def __init__(self, part1, part2):
        self.part1 = part1
        self.part2 = part2

    def get_partition_seg(self, segment_idx: int) -> List[str]:
        assert segment_idx == 0 or segment_idx == 1
        return [self.part1, self.part2][segment_idx]

    def get_first(self):
        return self.get_partition_seg(0)

    def get_second(self):
        return self.get_partition_seg(1)


class Segment1PartitionedPair(NamedTuple):
    segment1: PartitionedSegment
    segment2: List[str]
    pair_data: PairData

    def get_segment1_first(self):
        return self.segment1.get_first()

    def get_segment1_second(self):
        return self.segment1.get_second()


class Segment2PartitionedPair(NamedTuple):
    segment1: List[str]
    segment2: PartitionedSegment
    pair_data: PairData

    def get_segment2_first(self):
        return self.segment2.get_first()

    def get_segment2_second(self):
        return self.segment2.get_second()


class BothSegPartitionedPair(NamedTuple):
    segment1: PartitionedSegment
    segment2: PartitionedSegment
    pair_data: PairData

    def get(self, segment_no, part_no):
        segment = [self.segment1, self.segment2][segment_no]
        return segment.get(part_no)

    def get_segment1_first(self):
        return self.segment1.get_first()

    def get_segment1_second(self):
        return self.segment1.get_second()

    def get_segment2_first(self):
        return self.segment2.get_first()

    def get_segment2_second(self):
        return self.segment2.get_second()


def apply_segmentation_to_seg1(tokenizer, item: PairData) -> Segment1PartitionedPair:
    segment1_tokens = tokenizer.tokenize(item.segment1)
    segment2_tokens: List[str] = tokenizer.tokenize(item.segment2)
    st, ed = get_random_split_location(segment1_tokens)
    segment1 = RangePartitionedSegment(segment1_tokens, st, ed)
    return Segment1PartitionedPair(segment1, segment2_tokens, item)


class EvidencePair2(NamedTuple):
    query_like_segment: PartitionedSegment
    evidence_like_segment: PartitionedSegment


class BothSegPartitionedPairParser:
    def __init__(self, segment_len):
        self.segment_len = segment_len
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.wordpiece_tokenizer.vocab["[MASK]"]

    def parse(self, input_ids, segment_ids) -> EvidencePair2:
        input_ids = input_ids.numpy().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        segment_ids = segment_ids.numpy().tolist()

        tokens1: List[str] = tokens[:self.segment_len]
        segment_ids1 = segment_ids[:self.segment_len]

        tokens2 = tokens[self.segment_len:]
        segment_ids2 = segment_ids[self.segment_len:]

        def split(input_ids, segment_ids):
            seg2_start, seg2_end = get_st_ed(segment_ids)
            seg1 = input_ids[1: seg2_start-1]
            seg2 = input_ids[seg2_start: seg2_end-1]
            return seg1, seg2

        seg1_part1, seg2_part1 = split(tokens1, segment_ids1)
        seg1_part2, seg2_part2 = split(tokens2, segment_ids2)

        range_part_segs = [seg1_part1, seg1_part2]
        indice_part_segs = [seg2_part1, seg2_part2]
        if not len(indice_part_segs[0]) == len(indice_part_segs[1]):
            len1 = 3 + len(indice_part_segs[0]) + len(range_part_segs[0])
            len2 = 3 + len(indice_part_segs[1]) + len(range_part_segs[1])
            if len1 == self.segment_len or len2 == self.segment_len:
                pass
            else:
                c_log.warning("%d differs %d. ", len(indice_part_segs[0]), len(indice_part_segs[1]))
                c_log.warning("Length for other part : %d, %d", len(range_part_segs[0]), len(range_part_segs[1]))

        indices_part_merged: List[str] = self.merge_indices_part(indice_part_segs)
        mask_segment = MaskPartitionedSegment(*range_part_segs)

        def get_del_indices(p_i):
            indices = []
            for i, t in enumerate(p_i):
                if t == "[MASK]":
                    indices.append(i)
            return indices

        indices_segment = IndicesPartitionedSegment(
            indices_part_merged,
            get_del_indices(indice_part_segs[0]),
            get_del_indices(indice_part_segs[1]),
        )

        return EvidencePair2(mask_segment, indices_segment)

    def merge_indices_part(self, indice_part_segs: List[str]) -> List[str]:
        indices_part_merged = []
        for i, t in enumerate(indice_part_segs[0]):
            if t == "[MASK]":
                indices_part_merged.append(indice_part_segs[1][i])
            else:
                indices_part_merged.append(indice_part_segs[0][i])
        return indices_part_merged
