import os.path
import os.path
import sys

from omegaconf import OmegaConf

from misc_lib import path_join, BinHistogram
from tab_print import tab_print_dict
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import read_lines
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def main():
    voca_path = sys.argv[1]
    voca: List[str] = read_lines(voca_path)
    voca_d = {t: idx for idx, t in enumerate(voca)}
    save_dir = sys.argv[2]
    table_d = {}
    for q_term_i in range(len(voca)):
        log_path = path_join(save_dir, f"{q_term_i}.txt")
        if not os.path.exists(log_path):
            continue
        entry = load_table_as_indices(log_path, voca_d)
        table_d[q_term_i] = entry

    query = input("Wait: ")


def load_table_as_indices(log_path, voca_d):
    d_term_scores = [(row[0], float(row[1])) for row in tsv_iter(log_path)]
    entry = {}
    for d_term, score in d_term_scores:
        d_term_i = voca_d[d_term]
        entry[d_term_i] = score
    return entry


if __name__ == "__main__":
    main()
