import abc
from typing import List, Tuple

from data_generator.tokenizer_wo_tf import ids_to_text
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance


class PartialSegment:
    def __init__(self, data, n_seg):
        self.data = data
        self.n_seg = n_seg

    @classmethod
    def init_one_piece(cls, tokens: List[int]):
        return PartialSegment(tokens, 1)

    @classmethod
    def init_two_piece(cls, tokens: Tuple[List[int], List[int]]):
        return PartialSegment(tokens, 2)

    def to_text(self, tokenizer) -> str:
        if self.n_seg == 1:
            return ids_to_text(tokenizer, self.data)
        elif self.n_seg == 2:
            head, tail = self.data
            return ids_to_text(tokenizer, head) + " [MASK] " + ids_to_text(tokenizer, tail)
        else:
            raise Exception("n_seg > 2 is not expected")


class ComplementCandidateGenIF(abc.ABC):
    @abc.abstractmethod
    def get_candidates(self, si: SegmentedInstance, preserve_seg_idx) -> List[PartialSegment]:
        pass


class SearchComplementIF(abc.ABC):
    @abc.abstractmethod
    def search_complement(self, si: SegmentedInstance, preserve_seg_idx) -> List[List[int]]:
        pass


class SegJoinPolicyIF(abc.ABC):
    @abc.abstractmethod
    def join_tokens(self, si: SegmentedInstance, new_tokens: PartialSegment, preserve_seg_idx):
        pass
