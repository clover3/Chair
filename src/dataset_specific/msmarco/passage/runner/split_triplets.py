import gzip

from dataset_specific.msmarco.passage.path_helper import get_train_triples_path, MSMARCO_PASSAGE_TRIPLET_SIZE, \
    get_train_triples_partition_path
from misc_lib import TimeEstimator


def main():
    tsv_path = get_train_triples_path()

    ticker = TimeEstimator(
        MSMARCO_PASSAGE_TRIPLET_SIZE,
        sample_size=1000 * 10)

    n_per_partition = 1000 * 1000
    n_out = 0

    part_no = 0
    f = open(get_train_triples_partition_path(part_no), "w")
    for idx, line in enumerate(gzip.open(tsv_path, 'rt', encoding='utf8')):
        n_out += 1
        f.write(line)

        if n_out == n_per_partition:
            part_no += 1
            f.close()
            f = open(get_train_triples_partition_path(part_no), "w")
            n_out = 0

        ticker.tick()
    return NotImplemented


if __name__ == "__main__":
    main()