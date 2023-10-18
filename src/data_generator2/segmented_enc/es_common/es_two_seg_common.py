from abc import ABC, abstractmethod
from typing import NamedTuple
from typing import List


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
    # [CLS] segment1 [SEP] segment2 [SEP]  is expected
    segment1: PartitionedSegment
    segment2: PartitionedSegment
    pair_data: PairData

    @classmethod
    def from_seg1_partitioned_pair(cls, pair: Segment1PartitionedPair):
        return BothSegPartitionedPair(
            pair.segment1,
            IndicesPartitionedSegment(pair.segment2, [], []),
            pair.pair_data
        )

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


class EvidencePair2(NamedTuple):
    query_like_segment: PartitionedSegment
    evidence_like_segment: PartitionedSegment


class EvidencePair3(NamedTuple):
    query_like_segment: PartitionedSegment
    evidence_like_segment: IndicesPartitionedSegment


