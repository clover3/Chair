
from typing import List

import nltk

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementCandidateGenIF, \
    PartialSegment
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.problem_builder import get_word_level_location_w_ids


class ComplementGenOriginalQuery(ComplementCandidateGenIF):
    def get_candidates(self, si: SegmentedInstance, preserve_seg_idx) -> List[PartialSegment]:
        candidates: List[PartialSegment] = []
        if preserve_seg_idx == 0:
            seg_idx = 1
        elif preserve_seg_idx == 1:
            seg_idx = 0
        else:
            raise ValueError

        head_indices, tail_indices = si.text1.get_token_idx_as_head_tail(seg_idx)

        def get_seg1_tokens_by_indices(indices):
            return [si.text1.tokens_ids[i] for i in indices]

        head = get_seg1_tokens_by_indices(head_indices)
        tail = get_seg1_tokens_by_indices(tail_indices)

        candidates.append(PartialSegment((head, tail), 2))
        return candidates

