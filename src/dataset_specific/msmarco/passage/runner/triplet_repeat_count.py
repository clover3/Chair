from collections import Counter

from dataset_specific.msmarco.passage.path_helper import get_train_triples_partition_path
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from table_lib import tsv_iter


def main():
    def iter():
        for job_no in range(10):
            file_path = get_train_triples_partition_path(job_no)
            raw_train_iter: Iterable[tuple[str, str, str]] = tsv_iter(file_path)
            yield from raw_train_iter

    query_counter = Counter()
    d_p_counter = Counter()
    qd_p_counter = Counter()

    n_row = 0
    for q, dp, dn in iter():
        query_counter[q] += 1
        d_p_counter[dp] += 1
        qd_p_counter[(q, dp)] += 1
        n_row += 1

    print("Length of query_counter:", len(query_counter))
    print("Length of d_p_counter:", len(d_p_counter))
    print("Length of qd_p_counter:", len(qd_p_counter))

    print("Sum of query_counter values:", sum(query_counter.values()))
    print("Sum of values in d_p_counter:", sum(d_p_counter.values()))
    print("Sum of values in qd_p_counter:", sum(qd_p_counter.values()))

    print("Value of n_row:", n_row)

    return NotImplemented


if __name__ == "__main__":
    main()