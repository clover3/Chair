import os.path
import os.path
import sys

from omegaconf import OmegaConf

from misc_lib import path_join, BinHistogram
from tab_print import tab_print_dict
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import read_lines


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    q_terms = read_lines(conf.q_term_path)
    d_terms = read_lines(conf.d_term_path)
    table_save_path = conf.table_save_path
    constant_threshold = conf.constant_threshold
    save_dir = conf.score_save_dir
    num_items = len(q_terms)

    print("constant_threshold", constant_threshold)

    def bin_fn(n):
        interval = [0, 1, 5, 10, 50, 100, 500, 1000000]
        for i in range(len(interval) - 1):
            st = interval[i]
            ed = interval[i+1]
            if st <= n < ed:
                return "{} <= n < {}".format(st, ed)
        return "error"

    bin = BinHistogram(bin_fn)

    out_f = open(table_save_path, "w")
    n_done = 0
    n_term_pair = 0
    for q_term_i in range(len(q_terms)):
        log_path = path_join(save_dir, f"{q_term_i}.txt")
        if not os.path.exists(log_path):
            continue

        n_done += 1
        d_term_scores = [(row[0], float(row[1])) for row in tsv_iter(log_path)]
        q_term = q_terms[q_term_i]
        threshold = constant_threshold
        selected_terms = []
        for d_term, score in d_term_scores:
            if score >= threshold:
                if d_term != q_term:
                    selected_terms.append((d_term, score))

        bin.add(len(selected_terms))

        for d_term, score in selected_terms:
            out_f.write(f"{q_term}\t{d_term}\t{score}\n")

        n_term_pair += len(selected_terms)

    print("n_term_pair", n_term_pair)
    print("n_done", n_done)
    tab_print_dict(bin.counter)


if __name__ == "__main__":
    main()
