import random
import sys
from collections import Counter
from typing import Iterable, Tuple

from trainer_v2.per_project.transparency.misc_common import save_tsv


def tsv_iter(file_path) -> Iterable[Tuple]:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    for line in f:
        yield line.strip().split("\t")


def main():
    cut = 0.1
    table = tsv_iter(sys.argv[1])
    save_path = sys.argv[2]

    counter = Counter()
    pos_items = []
    neg_items = []

    for q, d, s in table:
        score = float(s)
        if score > cut:
            counter["pos"] += 1
            pos_items.append((q, d, s))

        if score < -cut:
            counter["neg"] += 1
            neg_items.append((q, d, s))

    print(counter)
    random.shuffle(neg_items)
    neg_items = neg_items[:len(pos_items)]

    out_table = pos_items + neg_items
    print(len(out_table))

    save_tsv(out_table, save_path)


if __name__ == "__main__":
    main()
