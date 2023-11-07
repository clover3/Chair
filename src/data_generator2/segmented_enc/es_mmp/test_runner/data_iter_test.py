import sys
from typing import Iterable, Tuple, List

from data_generator2.segmented_enc.es_mmp.data_iter_triplets import loosely_compare
from dataset_specific.msmarco.passage.path_helper import iter_train_triples_partition
from dataset_specific.msmarco.passage.processed_resource_loader import load_partitioned_query
from misc_lib import TimeEstimator
from taskman_client.wrapper3 import JobContext


def iter_qd(part_no):
    triplet_iter = iter_train_triples_partition(part_no)
    query_st_ed_iter: Iterable[Tuple[List[str], int, int]] = load_partitioned_query(part_no)

    data_idx = part_no * 1000000
    error_cnt = 0
    for (q, d1, d2), (q_sp_tokens, st, ed) in zip(triplet_iter, query_st_ed_iter):
        # check q equals q_tokens
        if data_idx % 17 == 1:
            if loosely_compare(q, q_sp_tokens):
                error_cnt = 0
            else:
                error_cnt += 1
                if error_cnt > 4:
                    print("query: ", q)
                    print("q_sp_tokens: ", q_sp_tokens)
                    raise ValueError()
        yield 0


def main():
    print("hi")
    with JobContext("Data Iter Test"):
        for part_no in range(397):
            print("Part", part_no)
            ticker = TimeEstimator(1000000)
            for _ in iter_qd(part_no):
                ticker.tick()



if __name__ == "__main__":
    main()
