
import os

from dataset_specific.msmarco.passage.path_helper import get_train_triples_small_path
from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join, batch_iter_from_entry_iter
from table_lib import tsv_iter
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample
from trainer_v2.per_project.transparency.mmp.data_gen.pair_train import get_encode_fn_for_pair_train


JOB_NAME = "mmp_train_pair_triplet_small"


def main():
    flush_size = 1000 * 1000
    save_dir = path_join("output", "msmarco", "passage", JOB_NAME)
    encode_fn = get_encode_fn_for_pair_train()

    part_no = 0
    itr = tsv_iter(get_train_triples_small_path())
    for batch_save in batch_iter_from_entry_iter(itr, flush_size):
        save_path = path_join(save_dir, str(part_no))
        write_records_w_encode_fn(save_path, encode_fn, batch_save)
        part_no += 1



if __name__ == "__main__":
    main()
