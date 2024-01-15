import csv
import sys

import pandas as pd
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import TimeEstimator
from trainer_v2.per_project.transparency.misc_common import save_tsv


def tsv_iter_here(file_path) -> Iterable[Tuple]:
    f = open(file_path, "r", encoding="utf-8", errors="ignore")
    for line in f:
        yield line.strip().split("\t")



def main():
    cut = 3.0
    table = tsv_iter_here(sys.argv[1])

    ticker = TimeEstimator(4049661)
    out_table = []
    for q, d, s in table:
        if float(s) > cut:
            out_table.append((q, d, s))
        ticker.tick()


    print(len(out_table))
    save_path = sys.argv[2]
    save_tsv(out_table, save_path)



if __name__ == "__main__":
    main()