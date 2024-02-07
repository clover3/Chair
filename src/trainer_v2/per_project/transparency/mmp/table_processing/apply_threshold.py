import os
import sys
from omegaconf import OmegaConf
from misc_lib import path_join, group_by
from table_lib import tsv_iter


def filter_by_threshold(conf):
    src_table_path = conf.src_table_path
    dst_table_path = conf.dst_table_path
    threshold = conf.threshold
    print(f"Keep term pairs which have higher than {threshold} score")

    out_f = open(dst_table_path, "w")
    n_before = 0
    n_after = 0

    for q_term, d_term, score_s in tsv_iter(src_table_path):
        n_before += 1
        if float(score_s) >= threshold:
            out_f.write(f"{q_term}\t{d_term}\t{score_s}\n")
            n_after += 1

    print(f"Reduced from {n_before} to {n_after}")


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    filter_by_threshold(conf)


if __name__ == "__main__":
    main()