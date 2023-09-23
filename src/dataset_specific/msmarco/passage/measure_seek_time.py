import gzip

from dataset_specific.msmarco.passage.path_helper import get_train_triples_path
from misc_lib import TimeEstimator


def main():
    tsv_path = get_train_triples_path()

    ticker = TimeEstimator(
        370 * 1000 * 1000,
        sample_size=1000 * 10)
    for line in gzip.open(tsv_path, 'rt', encoding='utf8'):
        ticker.tick()
    return NotImplemented


if __name__ == "__main__":
    main()