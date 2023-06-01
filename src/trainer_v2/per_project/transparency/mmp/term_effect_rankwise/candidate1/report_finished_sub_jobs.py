import os
import sys
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate1.run_candidate1 import \
    get_te_candidat1_job_group_proxy
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores


def check_done(q_term, d_term):
    job_no_itr = get_mmp_split_w_deep_scores()
    n_done = 0
    for job_no in job_no_itr:
        try:
            save_path = get_te_save_path(q_term, d_term, job_no)
            if os.path.exists(save_path):
                n_done += 1
        except FileNotFoundError:
            pass

    if len(job_no_itr) == n_done:
        return True
    else:
        print(f"out of {len(job_no_itr)} jobs {n_done} are actually loaded")
        return False


def main():
    proxy = get_te_candidat1_job_group_proxy()
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")

    todo_list = [line.strip() for line in open(save_path, "r")]
    for i in range(100, 210):
        q_term, d_term = todo_list[i].split()
        if check_done(q_term, d_term):
            proxy.sub_job_done(i)
            # print(q_term, d_term)
        else:
            print(f"Job {i} is not done")



if __name__ == "__main__":
    main()