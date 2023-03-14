import os
import sys

from misc_lib import path_join, TimeEstimator, exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.per_project.transparency.splade_regression.data_generator.encode_fns import get_vector_regression_encode_fn
from trainer_v2.per_project.transparency.splade_regression.data_loaders.regression_loader import VectorRegressionLoader
from trainer_v2.per_project.transparency.splade_regression.path_helper import partitioned_triplet_path_format_str
from trainer_v2.train_util.arg_flags import flags_parser
from typing import Iterable, Tuple


def main():
    args = flags_parser.parse_args("")
    text_path_format_str = partitioned_triplet_path_format_str()
    vector_dir = path_join("data", "splade", "splade_encode")
    max_partition = 5280
    max_seq_length = 256
    max_vector_indices = 512

    start_partition = int(sys.argv[1])
    end_partition = int(sys.argv[2])

    data_loader = VectorRegressionLoader(
        text_path_format_str, vector_dir,
        max_partition=max_partition)

    encode_fn = get_vector_regression_encode_fn(max_seq_length, max_vector_indices)
    save_dir = path_join("output", "splade", "splade_regression2")
    exist_or_mkdir(save_dir)

    num_partition = end_partition - start_partition
    ticker = TimeEstimator(num_partition)
    for partition in range(start_partition, end_partition):
        itr: Iterable[Tuple] = data_loader.iterate_tokenized(partition)
        save_path = os.path.join(save_dir, str(partition))
        write_records_w_encode_fn(save_path, encode_fn, itr)
        ticker.tick()


if __name__ == "__main__":
    main()
