import os
from pickle import UnpicklingError

import numpy as np
import logging
import sys
from typing import Iterable, Tuple

from cache import load_pickle_from
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.attn_compute.iter_attn import iter_attention_data_pair_as_pos_neg, \
    QDWithScoreAttn, get_attn2_save_dir
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition
from cpath import output_path
from misc_lib import path_join


def get_last_item(attn_save_dir, partition_no) -> Iterable[QDWithScoreAttn]:
    batch_no = 0
    last_valid = None
    while True:
        file_path = path_join(attn_save_dir, f"{partition_no}_{batch_no}")
        if os.path.exists(file_path):
            last_valid = file_path
        else:
            break
        batch_no += 1

    return last_valid



def verify(job_no: int):
    split = "train"
    c_log.setLevel(logging.DEBUG)

    attn_save_dir = get_attn2_save_dir()
    partition_todo = get_valid_mmp_partition(split)
    n_per_job = 10
    st = job_no * n_per_job
    ed = st + n_per_job
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue

        c_log.info("Partition %d", partition_no)
        # pos_neg_pair_itr: Iterable[Tuple[QDWithScoreAttn, QDWithScoreAttn]] = (
        #     iter_attention_data_pair_as_pos_neg(attn_save_dir, partition_no))

        file_path = get_last_item(attn_save_dir, partition_no)
        print("Last file: ", file_path)
        try:
            tuple_itr = load_pickle_from(file_path)
        except UnpicklingError as e:
            print("Unpickling error:", e)

        # for query, doc, score, attn in tuple_itr:
        #     w, h = item.attn.shape
        #     if not w or not h:
        #         print("Empty item:")
        #         print("query: ", item.query)
        #         print("doc: ", item.doc)
        #

def main():
    job_no = int(sys.argv[1])
    verify(job_no)


if __name__ == "__main__":
    main()
