from typing import List

from arg.qck.decl import QKUnit, KDP
from cache import load_from_pickle
from list_lib import lmap


def compare(qk1: List[QKUnit], qk2: List[QKUnit]):

    def get_qid_to_kdp_list(qk: List[QKUnit]):
        d = {}
        for q, k_list in qk:
            d[q.query_id] = k_list
        return d

    qk1_d = get_qid_to_kdp_list(qk1)
    qk2_d = get_qid_to_kdp_list(qk2)

    for qid in qk1_d:
        kdp_l1 = qk1_d[qid]
        if qid in qk2_d:
            kdp_l2 = qk2_d[qid]

            kdp_ids1: List[str] = lmap(KDP.to_str, kdp_l1)
            kdp_ids2: List[str] = lmap(KDP.to_str, kdp_l2)

            n_common = len(list([kid for kid in kdp_ids1 if kid in kdp_ids2]))

            n_remove = len(kdp_ids1) - n_common
            n_new = len(kdp_ids2) - n_common

            if n_remove or n_new:
                print(len(kdp_ids1), n_remove, n_new)
        else:
            print("qid not found:", qid)


def main():
    split = "train"
    qk1 = load_from_pickle("pc_qk2_filtered_{}".format(split))
    qk2 = load_from_pickle("pc_qk2_09_filtered_{}".format(split))
    compare(qk1, qk2)


if __name__ == "__main__":
    main()