import logging
import os
import pickle
import sys
from typing import List, Iterable, Tuple
from typing import Union

import numpy as np

from adhoc.misc_helper import enumerate_pos_neg_pairs, group_pos_neg
from cpath import at_output_dir
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData, BothSegPartitionedPair, \
    RangePartitionedSegment, IndicesPartitionedSegment
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import get_delete_indices_for_segment2_inner
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from list_lib import flatten
from misc_lib import exist_or_mkdir
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


# cur len = 3,
# cont_seg_len = 49
def build_get_num_delete_fn(del_rate: float):
    def get_num_delete(cur_part_len, other_part_len, cont_seg_len):
        # Assume each token of currrent part require one token in continuous one
        reasonable_max_del = cont_seg_len - cur_part_len
        reasonable_max_del = max(reasonable_max_del, 1)

        normal_mean = reasonable_max_del * del_rate
        std_dev = normal_mean
        num_del = int(np.random.normal(normal_mean, std_dev))

        # Cannot delete more than cont_seg_len
        num_del = min(cont_seg_len, num_del)
        num_del = max(0, num_del)
        return num_del
    return get_num_delete


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'data_generator2.segmented_enc.es_two_seg_common':
            module = 'data_generator2.segmented_enc.es_common.es_two_seg_common'
        return super().find_class(module, name)


PairWithAttn = Tuple[PairData, np.ndarray]


def generate_train_data(job_no: int, get_num_delete, dataset_name: str):
    output_dir = at_output_dir("tfrecord", dataset_name)
    split = "train"
    c_log.setLevel(logging.DEBUG)
    exist_or_mkdir(output_dir)
    partition_len = 256
    tokenizer = get_tokenizer()
    encoder = PartitionedEncoder(tokenizer, partition_len)
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

    def get_query_key(pair: PairData):
        return pair.segment1

    def group_iter(itr: Iterable[PairWithAttn]) -> Iterable[List[PairWithAttn]]:
        cur_group: List[PairWithAttn] = []
        cur_key: Union[str, None] = None
        for pair, attn in itr:
            key: Union[str, None] = get_query_key(pair)
            if key == cur_key:
                cur_group.append((pair, attn))
            else:
                if cur_group:
                    yield cur_group
                cur_group = [(pair, attn)]
                cur_key = key
        if cur_group:
            yield cur_group
            
    def is_pos(e: PairWithAttn):
        pair, attn = e
        return pair.label == "1"

    partition_todo = get_valid_mmp_partition(split)
    st = job_no
    ed = st + 1
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue
        save_path = os.path.join(output_dir, str(partition_no))

        c_log.info("Partition %d", partition_no)
        data_size = 30000
        attn_data_pair: Iterable[PairWithAttn] = iter_attention_data_pair(partition_no)

        grouped_itr: Iterable[List[PairWithAttn]] = group_iter(attn_data_pair)
        pos_neg_itr: Iterable[Tuple[List[PairWithAttn], List[PairWithAttn]]] = map(
            lambda e: group_pos_neg(e, is_pos), grouped_itr)
        pos_neg_pair_itr: Iterable[Tuple[PairWithAttn, PairWithAttn]] = flatten(map(
            enumerate_pos_neg_pairs, pos_neg_itr))

        def partition_sel_indices(e: Tuple[PairWithAttn, PairWithAttn]) -> \
                Tuple[BothSegPartitionedPair, BothSegPartitionedPair]:
            (pos_pair, pos_attn), (neg_pair, neg_attn) = e

            segment1_s: str = pos_pair.segment1
            assert pos_pair.segment1 == neg_pair.segment1

            segment1_tokens = tokenizer.tokenize(segment1_s)
            st, ed = get_random_split_location(segment1_tokens)
            partitioned_segment1: RangePartitionedSegment = RangePartitionedSegment(segment1_tokens, st, ed)

            def partition_pair(pair, attn_score) -> BothSegPartitionedPair:
                segment2_tokens: List[str] = tokenizer.tokenize(pair.segment2)
                delete_indices_list = get_delete_indices_for_segment2_inner(attn_score, partitioned_segment1, segment2_tokens, get_num_delete)
                partitioned_segment2 = IndicesPartitionedSegment(segment2_tokens, delete_indices_list[0], delete_indices_list[1])
                seg_pair = BothSegPartitionedPair(partitioned_segment1, partitioned_segment2, pair)
                return seg_pair

            return partition_pair(pos_pair, pos_attn), partition_pair(neg_pair, neg_attn)

        itr: Iterable[Tuple[BothSegPartitionedPair, BothSegPartitionedPair]] = map(partition_sel_indices, pos_neg_pair_itr)

        def encode_fn(t):
            a, b = t
            return encoder.encode_paired(a, b)

        write_records_w_encode_fn(save_path, encode_fn, itr, data_size)


def main():
    job_no = int(sys.argv[1])
    del_rate = 0.5
    dataset_name = "mmp_pep5"
    with JobContext(f"pep5_gen_{job_no}"):
        get_num_delete = build_get_num_delete_fn(del_rate)
        generate_train_data(job_no, get_num_delete, dataset_name)


if __name__ == "__main__":
    main()
