from collections import Counter

from contradiction.medical_claims.annotation_1.post_process2.join_by_long import join_from_binary_labels
from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.token_tagging.acc_eval.defs import SentTokenLabel


def get_pair_key_from_src2():
    counter = Counter()
    unique_pair_key: Set[Tuple[int, int]] = set()
    for tag in ["mismatch", "conflict"]:
        for split in ["val", "test"]:
            src2: List[SentTokenLabel] = load_sbl_binary_label(tag, split)
            for item in src2:
                group_no, inner_idx, _, _ = item.qid.split("_")
                pair_id = int(group_no), int(inner_idx)
                counter['sent'] += 1
                unique_pair_key.add(pair_id)
    print("Unique pairs", len(unique_pair_key))
    print(counter)
    return unique_pair_key


def main():
    unique_pair_key1: Set = join_from_binary_labels()
    unique_pair_key2: Set = get_pair_key_from_src2()

    print("From Set 1")
    for k in unique_pair_key1:
        if k not in unique_pair_key2:
            print(k)

    print("")
    print("From Set 2")
    for k in unique_pair_key2:
        if k not in unique_pair_key1:
            print(k)

    print(f"Set 1 has {len(unique_pair_key1)} keys")
    print(f"Set 2 has {len(unique_pair_key2)} keys")
    # Conclusion : Set1 is from only two annotators (Q / J)
    # Set 2 is from three annotators, where third annotator had 6 claim pair annotated


if __name__ == "__main__":
    main()
