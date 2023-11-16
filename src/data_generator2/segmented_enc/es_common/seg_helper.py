from data_generator.tokenizer_wo_tf import pretty_tokens
from data_generator2.segmented_enc.es_common.es_two_seg_common import RangePartitionedSegment


def seg_to_text(segment: RangePartitionedSegment, part_no: int) -> str:
    if part_no == 0:
        s1, s2 = segment.get_first()
        return pretty_tokens(s1, True) + " [MASK] " + pretty_tokens(s2, True)
    elif part_no == 1:
        seg = segment.get_second()
        return pretty_tokens(seg, True)
    else:
        raise Exception()
