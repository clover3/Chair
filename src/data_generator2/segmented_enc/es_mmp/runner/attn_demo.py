import itertools
import logging
import os
from typing import Iterable, Tuple, List

import numpy as np

from cpath import at_output_dir, output_path
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData, RangePartitionedSegment, \
    IndicesPartitionedSegment, BothSegPartitionedPair, PartitionedSegment
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import get_delete_indices_for_segment2_inner, \
    merge_attn_scores_for_partitions
from data_generator2.segmented_enc.es_common.partitioned_encoder import build_get_num_delete_fn
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttn, PairWithAttnEncoderIF
from data_generator2.segmented_enc.es_mmp.pep_attn_common import CustomUnpickler
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location, get_random_split_location2
from list_lib import foreach
from misc_lib import exist_or_mkdir, path_join
from tlm.token_utils import cells_from_tokens
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition
from visualize.html_visual import HtmlVisualizer, Cell


def get_evidence_score(
            attn_merged,
            segment1: PartitionedSegment,
            segment2: List[str],
            ):
    seg1_st = 1
    seg1_ed = seg1_st + len(segment1.tokens)
    seg2_st = seg1_ed + 1
    seg2_ed = seg2_st + len(segment2)
    # part_segment: The segment that is PARTitioned
    # cont_seg: The segment that is NOT partitioned, and thus CONTinuous
    assert isinstance(segment1, RangePartitionedSegment)
    part_segment: RangePartitionedSegment = segment1
    cont_seg_tokens = segment2
    part_seg_st = seg1_st
    part_seg_ed = seg1_ed
    cont_seg_st = seg2_st
    cont_seg_ed = seg2_ed
    part_seg_part_i_from_mean, part_seg_part_i_to_mean = \
        merge_attn_scores_for_partitions(attn_merged,
                                         cont_seg_st, cont_seg_ed,
                                         part_segment.st, part_segment.ed,
                                         part_seg_st, part_seg_ed)
    part_seg_part_0_mean = (part_seg_part_i_from_mean[0] + part_seg_part_i_to_mean[0]) / 2
    part_seg_part_1_mean = (part_seg_part_i_from_mean[1] + part_seg_part_i_to_mean[1]) / 2
    return part_seg_part_0_mean, part_seg_part_1_mean


class PrinterAttentionSelection:
    def __init__(self, get_num_delete, tokenizer, html_save_path):
        self.get_num_delete = get_num_delete
        self.tokenizer = tokenizer
        self.html = HtmlVisualizer(html_save_path)

    def print_pairwise_attn(self, item: PairWithAttn):
        # Print HTML
        # Partition item into two
        pair, attn_score = item

        segment1_s: str = pair.segment1

        segment1_tokens = self.tokenizer.tokenize(segment1_s)
        st, ed = get_random_split_location2(segment1_tokens)
        partitioned_segment1: RangePartitionedSegment = RangePartitionedSegment(segment1_tokens, st, ed)

        segment2_tokens: List[str] = self.tokenizer.tokenize(pair.segment2)
        delete_indices_list = get_delete_indices_for_segment2_inner(attn_score, partitioned_segment1,
                                                                    segment2_tokens, self.get_num_delete)
        partitioned_segment2 = IndicesPartitionedSegment(segment2_tokens, delete_indices_list[0],
                                                         delete_indices_list[1])
        seg_pair = BothSegPartitionedPair(partitioned_segment1, partitioned_segment2, pair)
        # For each query segment, highlight scores

        self.html.write_paragraph("Query: {}".format(pair.segment1))
        self.html.write_paragraph("Document: {}".format(pair.segment2))
        self.html.write_paragraph("label: {}".format(pair.label))

        scores1, scores2 = get_evidence_score(attn_score, partitioned_segment1, segment2_tokens)

        def normalize(s):
            try:
                return int(s * 1000)
            except ValueError as e:
                return 0

        scores_i = [scores1, scores2]
        for i in [0, 1]:
            scores = scores_i[i]
            query_seg = partitioned_segment1.get(i)
            if type(query_seg) == tuple:
                head, tail = query_seg
                s = head + ["[MASK]"] + tail
            else:
                s = query_seg

            query_s = pretty_tokens(s, True)
            self.html.write_paragraph("Query part: {}".format(query_s))

            assert len(scores) == len(segment2_tokens)
            try:
                norm_scores = [normalize(score) for score in scores]
            except ValueError as e:
                norm_scores = [0 for _ in scores]
                self.html.write_paragraph("NAN")

            def format_float(s):
                return f"{s:.2f}"[1:]

            row1 = cells_from_tokens(segment2_tokens, norm_scores)
            row2 = [Cell(format_float(s)) for s in scores]
            self.html.write_table([row1, row2])


def main():
    job_no = 0
    del_rate = 0.5
    get_num_delete = build_get_num_delete_fn(del_rate)
    tokenizer = get_tokenizer()
    printer = PrinterAttentionSelection(get_num_delete, tokenizer, "mmp1_attn.html")
    split = "train"
    c_log.setLevel(logging.DEBUG)
    attn_save_dir = path_join(output_path, "msmarco", "passage", "mmp1_attn")

    def iter_attention_data_pair(partition_no) -> Iterable[Tuple[PairData, np.array]]:
        batch_no = 0
        while True:
            file_path = path_join(attn_save_dir, f"{partition_no}_{batch_no}")
            if os.path.exists(file_path):
                c_log.info("Reading %s", file_path)
                f = open(file_path, "rb")
                obj = CustomUnpickler(f).load()
                attn_data_pair: List[Tuple[PairData, np.array]] = obj
                yield from attn_data_pair
            else:
                break
            batch_no += 1

    partition_todo = get_valid_mmp_partition(split)
    st = job_no
    ed = st + 1
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue
        c_log.info("Partition %d", partition_no)
        attn_data_pair: Iterable[PairWithAttn] = iter_attention_data_pair(partition_no)

        itr = itertools.islice(attn_data_pair, 100)
        foreach(printer.print_pairwise_attn, itr)
        break


if __name__ == "__main__":
    main()