import os
import pickle
from collections import Counter

from epath import job_man_dir


def load_doc_len():
    c = Counter()
    for job_id in range(40):
        counter = pickle.load(open(os.path.join(job_man_dir, "MMD_doc_len_cnt", str(job_id)), "rb"))
        c.update(counter)
    return c



def show():
    counter = load_doc_len()
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





def main():
    show()


if __name__ == "__main__":
    main()