import logging
import os
import sys
from typing import List, Iterable, Tuple

import numpy as np

from cache import load_pickle_from
from cpath import at_output_dir
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData, Segment1PartitionedPair, \
    BothSegPartitionedPair
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import compute_attn_sel_delete_indices
from data_generator2.segmented_enc.es_common.partitioned_encoder import apply_segmentation_to_seg1, PartitionedEncoder
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



def iter_attention_data_pair(attn_save_dir, partition_no) -> Iterable[Tuple[PairData, np.array]]:
    batch_no = 0
    while True:
        file_path = path_join(attn_save_dir, f"{partition_no}_{batch_no}")
        if os.path.exists(file_path):
            attn_data_pair: List[Tuple[PairData, np.array]] = load_pickle_from(file_path)
            yield from attn_data_pair
        else:
            break
        batch_no += 1


def generate_train_data(job_no: int, del_rate: float, dataset_name: str):
    output_dir = at_output_dir("tfrecord", dataset_name)
    split = "train"
    c_log.setLevel(logging.DEBUG)
    exist_or_mkdir(output_dir)
    partition_len = 256
    tokenizer = get_tokenizer()
    encoder = PartitionedEncoder(tokenizer, partition_len)
    encode_fn = encoder.encode

    attn_save_dir = path_join(output_path, "msmarco", "passage", "mmp1_attn")

    get_num_delete = build_get_num_delete_fn(del_rate)
    partition_todo = get_valid_mmp_partition(split)
    st = job_no
    ed = st + 1
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue
        save_path = os.path.join(output_dir, str(partition_no))
        if os.path.exists(save_path):
            continue

        c_log.info("Partition %d", partition_no)
        data_size = 30000
        attn_data_pair: Iterable[Tuple[PairData, np.array]] = iter_attention_data_pair(attn_save_dir, partition_no)

        def partition_sel_indices(e: Tuple[PairData, np.array]) -> BothSegPartitionedPair:
            pair_data, attn = e
            pair: Segment1PartitionedPair = apply_segmentation_to_seg1(tokenizer, pair_data)
            b_pair: BothSegPartitionedPair = compute_attn_sel_delete_indices(pair, attn, get_num_delete)
            return b_pair

        itr: Iterable[BothSegPartitionedPair] = map(partition_sel_indices, attn_data_pair)
        write_records_w_encode_fn(save_path, encode_fn, itr, data_size)


def main():
    job_no = int(sys.argv[1])
    del_rate = 0.5
    dataset_name = "mmp_pep4"
    with JobContext(f"pep4_gen_{job_no}"):
        generate_train_data(job_no, del_rate, dataset_name)


if __name__ == "__main__":
    main()
