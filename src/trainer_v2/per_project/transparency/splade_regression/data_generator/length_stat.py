import os
from collections import Counter

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
    max_partition = 10
    data_loader = VectorRegressionLoader(
        text_path, vector_dir,
        max_partition=max_partition,
        run_config=run_config)
    counter = Counter()
    itr: Iterable[Tuple] = data_loader.iterate_tokenized()
    for item in itr:
        x, y = item
        input_ids, _ = x

        counter[len(input_ids)] += 1



    n_all = sum(counter.values())

    keys = list(counter.keys())
    keys.sort()

    acc = 0
    for key in keys:
        acc += counter[key]
        percent = acc / n_all
        print("L <= {0} : {1:.2f}".format(key, percent))


    return NotImplemented


if __name__ == "__main__":
    main()