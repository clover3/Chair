from data_generator.tokenizer_wo_tf import pretty_tokens
from data_generator2.segmented_enc.es_common.es_two_seg_common import RangePartitionedSegment, BothSegPartitionedPair, \
    compress_mask
from data_generator2.segmented_enc.es_common.seg_helper import seg_to_text
from misc_lib import two_digit_float
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.runner.ee_train_gen import iter_pep_inference


def main():
    itr = iter_pep_inference(0)
    cnt = 0
    for item in itr:
        candidates, scores = item
        first_c = BothSegPartitionedPair(*candidates[0])

        if not isinstance(first_c.segment1, RangePartitionedSegment):
            raise TypeError("pair.segment1 should be RangePartitionedSegment")

        segment1: RangePartitionedSegment = first_c.segment1
        for part_no in [0, 1]:
            print(f"Query seg {part_no + 1}", seg_to_text(segment1, part_no))

        for i, c in enumerate(candidates):
            seq: list[str] = BothSegPartitionedPair(*c).segment2.get_first()
            seq: list[str] = compress_mask(seq)
            print("{}\t{}\t{}".format(
                pretty_tokens(seq, True),
                two_digit_float(scores[i][0]),
                two_digit_float(scores[i][1])))

        print()

        cnt += 1
        if cnt > 10:
            break


if __name__ == "__main__":
    main()