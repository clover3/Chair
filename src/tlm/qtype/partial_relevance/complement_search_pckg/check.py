from typing import Callable, List

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import SegJoinPolicyIF, \
    PartialSegment


class CheckComplementCandidate:
    def __init__(self,
                 forward_fn: Callable[[List[SegmentedInstance]], List[float]],
                 join_policy: SegJoinPolicyIF,
                 ):
        self.tokenizer = get_tokenizer()
        self.seg_join_policy: SegJoinPolicyIF = join_policy
        self.forward_fn = forward_fn

    def check_complement_list(self, si: SegmentedInstance, seg_index_to_keep, candidates: List[PartialSegment])\
            -> List[PartialSegment]:
        payload_list: List[SegmentedInstance] = self.get_payload_list(si, seg_index_to_keep, candidates)
        scores: List[float] = self.forward_fn(payload_list)

        def is_valid(s):
            return s > 0.5

        is_valid_list = map(is_valid, scores)

        for new_si, s in zip(payload_list, scores):
            if s > 0.5:
                print(s, self.tokenizer.convert_ids_to_tokens(new_si.text1.tokens_ids))

        complement_list: List[PartialSegment] = []
        for c, is_valid in zip(candidates, is_valid_list):
            if is_valid:
                complement_list.append(c)
        return complement_list

    def get_payload_list(self, si: SegmentedInstance, seg_index_to_keep, candidates: List[PartialSegment])\
            -> List[SegmentedInstance]:
        payload_list: List[SegmentedInstance] = []
        for c in candidates:
            new_text1 = self.seg_join_policy.join_tokens(si.text1, c, seg_index_to_keep)
            new_si: SegmentedInstance = SegmentedInstance(new_text1, si.text2)
            payload_list.append(new_si)
        return payload_list

    def get_scores(self, si: SegmentedInstance, seg_index_to_keep, candidates: List[PartialSegment]):
        payload_list = self.get_payload_list(si, seg_index_to_keep, candidates)
        scores: List[float] = self.forward_fn(payload_list)
        return scores