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
    counter: Counter = get_doc_length_counter_from_pickle()
    seg_length = 500

    all_keys = list(counter.keys())
    all_keys.sort()

    num_seg_count = Counter()
    for l in all_keys:
        num_seg = ceil_divide(l, seg_length)
        cnt = counter[l]
        assert type(cnt) == int
        num_seg_count[num_seg] += cnt

    num_docs = sum(counter.values())
    acc_portion = 0
    for key in sorted(num_seg_count.keys()):
        cnt = num_seg_count[key]
        assert type(cnt) == int
        portion = cnt / num_docs
        acc_portion += portion
        # print("{0}\t{1}\t{2:.2f}\t{3:.2f}".format(key, cnt, portion, acc_portion))
        print("{0}\t{1}\t{2:.4f}\t{3:.4f}".format(key, cnt, portion, acc_portion))


def show_median_90_percentile():
    counter: Counter = get_doc_length_counter_from_pickle()
    num_docs = sum(counter.values())

    percentiles = list([i/10 for i in range(10)])
    percentile_idx = 0
    print("num_docs", num_docs)

    cnt_acc = 0
    for doc_len in range(10000):
        if doc_len in counter:
            cnt = counter[doc_len]
            cnt_acc += cnt
            percentile = cnt_acc / num_docs
            if percentile > percentiles[percentile_idx]:
                print("percentile goal={} cur percentile={} cnt_acc={} doc_len={}".
                      format(percentiles[percentile_idx], percentile, cnt_acc, doc_len))
                percentile_idx += 1





if __name__ == "__main__":
    show_median_90_percentile()