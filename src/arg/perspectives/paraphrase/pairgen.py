import random
from itertools import combinations
from typing import List, Iterable, Dict, Tuple, Iterator, NamedTuple

# Ratio
# Same cluster(positive) : 1
# Same claim, different cluster (negative) : 3
# Different claim (negative) : 6
from arg.perspectives.load import load_claim_ids_for_split, get_claim_perspective_id_dict
from list_lib import flatten


# TODO : Load all clusters for each claim


class Instance(NamedTuple):
    pid1: int
    pid2: int
    label: int


def generate_pair_insts(split) -> Iterable[Instance]:
    pos_rate = 1
    neg1_rate = 3
    neg2_rate = 6
    ids: List[int] = list(load_claim_ids_for_split(split))
    id_dict: Dict[int, List[List[int]]] = get_claim_perspective_id_dict()

    def same_cluster_example() -> Iterator[Tuple[int, int]]:
        for claim_id in ids:
            clusters = id_dict[claim_id]
            for cluster in clusters:
                for p1, p2 in combinations(cluster, 2):
                    yield p1, p2

    def same_claim_different_cluster() -> Iterator[Tuple[int, int]]:
        for claim_id in ids:
            clusters = id_dict[claim_id]
            for cluster1, cluster2 in combinations(clusters, 2):
                for p1 in cluster1:
                    for p2 in cluster2:
                        yield p1, p2

    def different_claim() -> Iterator[Tuple[int, int]]:
        for cid1, cid2 in combinations(ids, 2):
            clusters1 = id_dict[cid1]
            clusters2 = id_dict[cid2]
            for p1 in flatten(clusters1):
                for p2 in flatten(clusters2):
                    yield p1, p2

    pos: List[Tuple[int, int]] = list(same_cluster_example())
    neg1: List[Tuple[int, int]] = list(same_claim_different_cluster())
    neg2: List[Tuple[int, int]] = list(different_claim())

    pos_len = len(pos)
    neg1_len = pos_len * neg1_rate
    neg2_len = pos_len * neg2_rate

    print("pos/neg1/neg2 = {}/{}/{}".format(pos_len, neg1_len, neg2_len))

    random.shuffle(neg1)
    random.shuffle(neg2)

    neg1 = neg1[:neg1_len]
    neg2 = neg2[:neg2_len]

    pos_data = list([Instance(pid1, pid2, 1) for pid1, pid2 in pos])
    neg_data = list([Instance(pid1, pid2, 0) for pid1, pid2 in neg1 + neg2])

    all_data = pos_data + neg_data
    random.shuffle(all_data)
    return all_data


if __name__ == "__main__":
    generate_pair_insts("train")
