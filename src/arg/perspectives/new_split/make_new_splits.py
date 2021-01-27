import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from cpath import data_path
from evals.parse import load_qrels_flat
from list_lib import lmap


def split_qids():
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

    def pool_qids(remaining_qids: List[str], seen_qids: Set[str], n_minimum: float):
        selected_qids = []
        idx = 0
        while len(selected_qids) < n_minimum and idx < len(remaining_qids):
            cur_qid = remaining_qids[idx]
            if cur_qid not in seen_qids:
                qids_to_add = [cur_qid]
                seen_qids.add(cur_qid)

                while qids_to_add:
                    qid_being_added = qids_to_add[0]
                    selected_qids.append(qid_being_added)
                    qids_to_add = qids_to_add[1:]
                    rel_qids = rel_qid_d[qid_being_added]
                    for qid in rel_qids:
                        if qid not in seen_qids:
                            qids_to_add.append(qid)
                            seen_qids.add(qid)

            idx += 1
        return selected_qids, remaining_qids[idx:]

    seen_qids = set()
    train_qids, remaining_qids = pool_qids(all_qids, seen_qids, min_train_claim)
    test_qids, remaining_qids = pool_qids(remaining_qids, seen_qids, min_test_claim)
    dev_qids = list([qid for qid in remaining_qids if qid not in seen_qids])

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


def get_similar_pairs(all_qids, qrels):
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

    def get_dict(qid_list, split_name) -> Dict[str, str]:
        d = {}
        for qid in qid_list:
            d[qid] = split_name
        return d

    d: Dict[str, str] = {}

    d.update(get_dict(train, "train"))
    d.update(get_dict(dev, "dev"))
    d.update(get_dict(test, "test"))

    assert False and "This routine should not be executed"
    save_path = os.path.join(data_path, "perspective", "new_split.json")
    json.dump(d, open(save_path, "w"))


if __name__ == "__main__":
    main()



