import os
import sys
from misc_lib import path_join
from table_lib import tsv_iter
from taskman_client.wrapper3 import JobContext
from trainer_v2.per_project.transparency.mmp.pep_to_tt.mct4.score_mct4 import build_mct4_config


def run_combination_filtering(conf):
    table_save_path = conf.table_save_path
    constant_threshold = conf.constant_threshold
    save_dir = conf.score_save_dir

    print("constant_threshold", constant_threshold)
    out_f = open(table_save_path, "w")
    n_done = 0
    n_term_pair = 0
    for job_idx in range(conf.num_jobs):
        log_path = path_join(save_dir, f"{job_idx}.txt")
        if not os.path.exists(log_path):
            continue

        n_done += 1
        selected_terms = []
        for q_term, d_term, score_s in tsv_iter(log_path):
            score = float(score_s)
            if score >= constant_threshold:
                selected_terms.append((q_term, d_term, score))

        for q_term, d_term, score in selected_terms:
            out_f.write(f"{q_term}\t{d_term}\t{score}\n")

        n_term_pair += len(selected_terms)
    print("n_term_pair", n_term_pair)
    print("n_done", n_done)


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    conf = build_mct4_config(model_name, step)
    task_name = conf.job_name_base + "_combine"
    with JobContext(task_name):
        run_combination_filtering(conf)


if __name__ == "__main__":
    main()
