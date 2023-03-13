import os
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.per_project.transparency.splade_regression.data_generator.encode_fns import get_vector_regression_encode_fn
from trainer_v2.per_project.transparency.splade_regression.data_loaders.regression_loader import VectorRegressionLoader
from trainer_v2.train_util.arg_flags import flags_parser
from typing import Iterable, Tuple


def main():
    args = flags_parser.parse_args("")
    run_config = get_run_config2(args)
    text_path = path_join("data", "msmarco", "splade_triplets", "raw.tsv")
    vector_dir = path_join("data", "splade", "splade_encode")
    max_partition = 1
    data_loader = VectorRegressionLoader(
        text_path, vector_dir,
        max_partition=max_partition,
        run_config=run_config)
    itr: Iterable[Tuple] = data_loader.iterate_tokenized()
    save_dir = path_join("output", "splade", "regression_tfrecord_ub")
    max_seq_length = 256
    save_path = os.path.join(save_dir, "one.tfrecord")
    encode_fn = get_vector_regression_encode_fn(max_seq_length)
    n_item = int(10000 * 3) * max_partition
    write_records_w_encode_fn(save_path, encode_fn, itr, n_item)




if __name__ == "__main__":
    main()
