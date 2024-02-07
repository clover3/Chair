import os
import sys
from omegaconf import OmegaConf
from misc_lib import path_join, group_by
from table_lib import tsv_iter


def filter_frequent_q_terms(conf):
    src_table_path = conf.src_table_path
    dst_table_path = conf.dst_table_path
    max_num = conf.max_num
    print(f"Drop query terms which has more than {max_num} entries")

    out_f = open(dst_table_path, "w")
    n_before = 0
    n_after = 0

    grouped = group_by(tsv_iter(src_table_path), lambda x: x[0])
    excluded_terms = []
    for q_term, entries in grouped.items():
        n_before += len(entries)
        if len(entries) > max_num:
            excluded_terms.append(q_term)
        else:
            n_after += len(entries)
            for q_term, d_term, score_s in entries:
                out_f.write(f"{q_term}\t{d_term}\t{score_s}\n")

    print(f"Reduced from {n_before} to {n_after}")


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    filter_frequent_q_terms(conf)


if __name__ == "__main__":
    main()