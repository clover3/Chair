from dataset_specific.msmarco.passage.grouped_reader import get_train_neg5_sample_path
from table_lib import tsv_iter


def main():
    partition_no = 1
    itr = tsv_iter(get_train_neg5_sample_path(partition_no))
    for query, doc, label in itr:
        pass
    return NotImplemented


if __name__ == "__main__":
    main()

