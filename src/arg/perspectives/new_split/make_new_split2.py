import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from cpath import data_path
from evals.parse import load_qrels_flat
from list_lib import lmap, lflatten


def make_clusters(all_qids: List[str], rel_qid_d: Dict[str, List[str]]) -> List[List[str]]:
    seen_qids = set()
    cluster_list = []
    idx = 0
    while idx < len(all_qids):
        cur_qid = all_qids[idx]
        if cur_qid not in seen_qids:
            qids_to_add = [cur_qid]
            queue_idx = 0
            seen_qids.add(cur_qid)

            while queue_idx < len(qids_to_add):
                qid_being_added = qids_to_add[queue_idx]
                rel_qids = rel_qid_d[qid_being_added]
                for qid in rel_qids:
                    if qid not in seen_qids:
                        qids_to_add.append(qid)
                        seen_qids.add(qid)
                queue_idx += 1
            cluster_list.append(qids_to_add)
        idx += 1
    return cluster_list


def split_qids() -> Tuple[List[str], List[str], List[str]]:
    judgment_path = os.path.join(data_path, "perspective", "qrel.txt")
    qrels = load_qrels_flat(judgment_path)

    all_qids = list(qrels.keys())
    n_qids = len(all_qids)

    paired_qids = get_similar_pairs(all_qids, qrels)

    rel_qid_d = defaultdict(list)
    for qid1, qid2 in paired_qids:
        rel_qid_d[qid1].append(qid2)
        rel_qid_d[qid2].append(qid1)

    random.shuffle(all_qids)
    n_claim = len(all_qids)
    # Previous split size
    # 'train': 541,
    # 'dev': 139,
    # 'test': 227,
    min_train_claim = 541
    min_test_claim = 227

    clusters: List[List[str]] = make_clusters(all_qids, rel_qid_d)
    random.shuffle(clusters)

    def pool_cluster(remaining_clusters: List[List[str]], start_idx, n_minimum):
        idx = start_idx
        selected_qids = []
        while idx < len(remaining_clusters) and len(selected_qids) < n_minimum:
            cur_cluster = remaining_clusters[idx]
            print(len(cur_cluster), end=" ")
            selected_qids.extend(cur_cluster)
            idx += 1
        print()
        return selected_qids, idx

    print("train")
    train_qids, last_idx = pool_cluster(clusters, 0, min_train_claim)
    print("test")
    test_qids, last_idx = pool_cluster(clusters, last_idx, min_test_claim)
    dev_qids = lflatten(clusters[last_idx:])
    qid_splits = [train_qids, test_qids, dev_qids]

    assert n_qids == sum(map(len, qid_splits))

    def overlap(l1: List, l2:List) -> Set:
        common = set(l1).intersection(l2)
        return common

    assert not overlap(train_qids, test_qids)
    assert not overlap(train_qids, dev_qids)
    assert not overlap(dev_qids, test_qids)
    for split_qid in qid_splits:
        assert len(set(split_qid)) == len(split_qid)

    return train_qids, dev_qids, test_qids


def get_similar_pairs(all_qids, qrels) -> List[Tuple[str, str]]:
    claim_to_perpsective: Dict[str, Set] = {}
    for qid, gold_list in qrels.items():
        true_gold: List[str] = list([doc_id for doc_id, score in gold_list if score > 0])
        claim_to_perpsective[qid] = set(true_gold)

    def overlap_score(qid1: str, qid2: str):
        doc_ids1: Set = claim_to_perpsective[qid1]
        doc_ids2: Set = claim_to_perpsective[qid2]

        if not doc_ids1 or not doc_ids2:
            return 0

        common = doc_ids1.intersection(doc_ids2)
        n_common = len(common)

        s1 = n_common / len(doc_ids1)
        s2 = n_common / len(doc_ids2)
        return max(s1, s2)

    count = 0
    paired_qids: List[Tuple[str, str]] = []
    for qid1 in all_qids:
        for qid2 in all_qids:
            if not int(qid1) < int(qid2):
                continue
            score = overlap_score(qid1, qid2)
            if score > 0.5:
                count += 1
                paired_qids.append((qid1, qid2))
    print("{} claims".format(len(all_qids)))
    print("{} claim pairs are similar".format(count))
    return paired_qids


def main():
    train, dev, test = split_qids()
    print("split size (train/dev/test)")
    print(lmap(len, [train, dev, test]))

    def get_dict(qid_list: List[str], split_name: str) -> Dict[str, str]:
        d = {}
        for qid in qid_list:
            d[qid] = split_name
        return d

    d: Dict[str, str] = {}
    d.update(get_dict(train, "train"))
    d.update(get_dict(dev, "dev"))
    d.update(get_dict(test, "test"))

    save_path = os.path.join(data_path, "perspective", "new_split2.json")
    json.dump(d, open(save_path, "w"))


if __name__ == "__main__":
    split_qids()
    # main()



