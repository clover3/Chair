from collections import Counter
from typing import List, Dict

from cache import load_from_pickle
from misc_lib import ceil_divide
from tlm.robust.load import load_robust_tokens


def get_doc_length_counter():
    tokens_d: Dict[str, List[str]] = load_robust_tokens()
    counter = Counter()
    for key, tokens in tokens_d.items():
        counter[len(tokens)] += 1
    return counter

def get_doc_length_counter_from_pickle():
    return load_from_pickle("robust_doc_length_counter")



def main():
    # counter = get_doc_length_counter()
    # save_to_pickle(counter, "robust_doc_length_counter")
    counter = get_doc_length_counter_from_pickle()
    seg_length = 500

    all_keys = list(counter.keys())
    all_keys.sort()

    num_seg_count = Counter()
    for l in all_keys:
        num_seg = ceil_divide(l, seg_length)
        cnt = counter[l]
        num_seg_count[num_seg] += cnt

    num_docs = sum(counter.values())
    acc_portion = 0
    for key in sorted(num_seg_count.keys()):
        cnt = num_seg_count[key]
        portion = cnt / num_docs
        acc_portion += portion
        print("{0}\t{1}\t{2:.2f}\t{3:.2f}".format(key, cnt, portion, acc_portion))


if __name__ == "__main__":
    main()