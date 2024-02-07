import os
import sys

from table_lib import tsv_iter
from taskman_client.wait_job_group import wait_job_group
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep_to_tt.mct6.score_mct6 import build_mct6_config
from cpath import output_path
from misc_lib import path_join


def run_combination_filtering(conf):
    table_save_path = conf.table_save_path
    constant_threshold = conf.constant_threshold
    save_dir = conf.score_save_dir

    print("constant_threshold", constant_threshold)
    out_f = open(table_save_path, "w")
    n_done = 0
    n_term_pair = 0
    q_terms = read_lines(conf.q_term_path)
    n_q_term_with_entry = 0
    for job_idx, q_term in enumerate(q_terms):
        log_path = path_join(save_dir, f"{job_idx}.txt")
        if not os.path.exists(log_path):
            continue

        n_done += 1
        selected_terms = []
        for d_term, score_s in tsv_iter(log_path):
            score = float(score_s)
            if score >= constant_threshold:
                selected_terms.append((q_term, d_term, score))

        for q_term, d_term, score in selected_terms:
            out_f.write(f"{q_term}\t{d_term}\t{score}\n")

        if selected_terms:
            n_q_term_with_entry += 1
        n_term_pair += len(selected_terms)
    print("n_term_pair", n_term_pair)
    print("n_done", n_done)
    print("n_q_term_with_entry", n_q_term_with_entry)


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    conf = build_mct6_config(model_name, step)
    task_name = conf.job_name_base + "_combine"

    with JobContext(task_name):
        c_log.info("Waiting for job group %s ", conf.job_name_base)
        wait_job_group(conf.job_name_base)
        c_log.info("Waiting Done")
        run_combination_filtering(conf)


if __name__ == "__main__":
    main()
