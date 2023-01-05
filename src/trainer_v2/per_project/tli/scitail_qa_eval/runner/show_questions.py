from collections import Counter

from dataset_specific.scitail import get_scitail_questions, load_scitail_structured
from misc_lib import BinHistogram, pause_hook


def count_qs():
    counter = Counter()
    e_list = load_scitail_structured("train")
    questions = [e.question for e in e_list]

    for q in questions:
        counter[q] += 1

    n_counter = Counter()
    def bin_fn(n):
        if n < 10:
            return n
        else:
            return n - n % 10

    bin = BinHistogram(bin_fn)
    for key, cnt in counter.items():
        n_counter[cnt] += 1
        bin.add(cnt)

    keys = list(bin.counter.keys())
    keys.sort()
    for k in keys:
        print(k, bin.counter[k])


def show_q():
    e_list = load_scitail_structured("train")
    for e in pause_hook(e_list, 100):
        print(e.question)


if __name__ == "__main__":
    show_q()