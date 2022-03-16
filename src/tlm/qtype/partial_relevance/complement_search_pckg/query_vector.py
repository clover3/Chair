from typing import List

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.analysis_fde.fde_module import FDEModule
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementCandidateGenIF, \
    PartialSegment


def to_id_format(tokenizer, s) -> PartialSegment:
    head, tail = s.split("[MASK]")

    def convert(s) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

    data = convert(head), convert(tail)
    return PartialSegment(data, 2)


class ComplementGenByQueryVector(ComplementCandidateGenIF):
    def __init__(self, fde_module: FDEModule):
        self.tokenizer = get_tokenizer()
        self.fde_module: FDEModule = fde_module

    def get_candidates(self, si: SegmentedInstance, preserve_seg_idx) -> List[PartialSegment]:
        text1_preserve_seg: List[int] = si.text1.get_tokens_for_seg(preserve_seg_idx)
        func_spans: List[str] = self.fde_module.get_promising_from_ids(text1_preserve_seg, si.text2.tokens_ids)

        def to_id_format_fn(func_span):
            return to_id_format(self.tokenizer, func_span)

        candidates: List[PartialSegment] = list(map(to_id_format_fn, func_spans))
        return candidates


