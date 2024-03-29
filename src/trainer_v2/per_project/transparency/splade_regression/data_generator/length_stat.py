from collections import Counter

from misc_lib import path_join, print_length_stats_from_counters
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.per_project.transparency.splade_regression.data_loaders.regression_loader import VectorRegressionLoader
from trainer_v2.train_util.arg_flags import flags_parser
from typing import Iterable, Tuple



def main():
    print("Length stat for vectors")
    args = flags_parser.parse_args("")
    run_config = get_run_config2(args)
    text_path = path_join("data", "msmarco", "splade_triplets", "raw.tsv")
    vector_dir = path_join("data", "splade", "splade_encode")
    max_partition = 10
    data_loader = VectorRegressionLoader(text_path, vector_dir, max_partition=max_partition)
    counter = Counter()
    itr: Iterable[Tuple] = data_loader.iterate_tokenized()
    for item in itr:
        x, y = item
        # input_ids, _ = x
        indices, values = y
        assert len(indices) == 1
        indices = indices[0]

        assert len(indices) == len(values)
        counter[len(indices)] += 1
    print_length_stats_from_counters(counter)

    return NotImplemented


if __name__ == "__main__":
    main()