import itertools
import sys

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_gen.pair_train import get_encode_fn_for_pointwise
from cpath import output_path
from misc_lib import path_join, select_third_fourth
from typing import List, Iterable, Callable, Dict, Tuple, Set



def main():
    job_no = sys.argv[1]
    quad_tsv_path = path_join("data", "msmarco", "passage", "group_sorted_10K", str(job_no))
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    save_dir = path_join("output", "msmarco", "passage", "mmp_train_split_all_tfrecord")
    save_path = path_join(save_dir, str(job_no))
    encode_fn = get_encode_fn_for_pointwise()
    n_item = 4234340
    # n_item = 1000 * 100
    write_records_w_encode_fn(save_path, encode_fn, tuple_itr, n_item)


if __name__ == "__main__":
    main()