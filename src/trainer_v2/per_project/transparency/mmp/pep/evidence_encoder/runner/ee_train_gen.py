import os
import sys

import numpy as np

from cache import load_pickle_from
from cpath import output_path, at_output_dir
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, compress_mask
from data_generator2.segmented_enc.es_common.partitioned_encoder import segment_formatter2
from misc_lib import path_join, SuccessCounter, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from typing import List, Iterable

from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.encoder_util import MultipleQDSegmentsEncoder

InferencePerQD = tuple[list[BothSegPartitionedPair], list[np.array]]


def iter_pep_inference(part_no: int) -> Iterable[InferencePerQD]:
    save_dir = path_join(output_path, "mmp", "pep10_del")
    batch_no = 0
    while True:
        file_path = path_join(save_dir, f"{part_no}_{batch_no}")
        if os.path.exists(file_path):
            c_log.info("Reading %s", file_path)
            obj = load_pickle_from(file_path)
            items: List[InferencePerQD] = obj
            yield from items
        else:
            break
        batch_no += 1


# Each QD makes
#   2 query segments
#   M document segments
#   2 * M scores
def encode_per_qd(suc: SuccessCounter, max_seq_length, e: InferencePerQD)\
        -> tuple[list[list], list[list],  np.ndarray]:
    candidates: list[BothSegPartitionedPair] = e[0]
    scores: list[np.array] = e[1]

    first_c = BothSegPartitionedPair(*candidates[0])
    query_seg = first_c.segment1
    q_seq_list: list[list[str]] = []

    for part_no in [0, 1]:
        sequence: list[str] = segment_formatter2(query_seg, part_no)
        if len(sequence) + 1 < max_seq_length:
            pass
        else:
            sequence = []
        q_seq_list.append(sequence)

    dt_seq_list: list[list[str]] = []
    for c in candidates:
        seq: list[str] = BothSegPartitionedPair(*c).segment2.get_first()
        seq: list[str] = compress_mask(seq)
        if len(seq) + 1 < max_seq_length:
            pass
        else:
            sequence = []
        dt_seq_list.append(sequence)

    for q_part_no in [0, 1]:
        for dt_idx in range(len(candidates)):
            if not q_seq_list[q_part_no] or not dt_seq_list[dt_idx]:
                suc.fail()
                scores[dt_idx][q_part_no] = 0
            else:
                suc.suc()

    scores_np = np.array(scores)
    scores_np = np.transpose(scores_np, [1, 0])
    return q_seq_list, dt_seq_list, scores_np


# Objective

def main():
    dataset_name = "mmp_ee"
    output_dir = at_output_dir("tfrecord", dataset_name)
    exist_or_mkdir(output_dir)

    job_no = int(sys.argv[1])
    suc = SuccessCounter()
    seq_len = 32
    max_qt = 2
    max_dt = 20
    encoder = MultipleQDSegmentsEncoder(seq_len, max_qt, max_dt)

    st = job_no * 10
    ed = st + 10
    for part_no in range(st, ed):
        itr = iter_pep_inference(part_no)
        save_path = path_join(output_dir, str(part_no))
        def encode_fn(item):
            q_seq_list, dt_seq_list, scores = encode_per_qd(suc, seq_len, item)
            ordered_dict = encoder.pack_to_order_dict(q_seq_list, dt_seq_list, scores)
            return ordered_dict

        write_records_w_encode_fn(save_path, encode_fn, itr)


if __name__ == "__main__":
    main()
