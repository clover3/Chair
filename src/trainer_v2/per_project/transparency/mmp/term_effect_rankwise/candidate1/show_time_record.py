import csv
import os

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path_base
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.summarize_te import get_score


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")
    todo_list = [line.strip() for line in open(cand_path, "r")]
    base_time = None
    for i in range(0, 10000, 100):
        q_term, d_term = todo_list[i].split()
        first_job_no = 0
        last_job_no = 118
        def get_m_time(job_no):
            nonlocal base_time
            save_path = get_te_save_path_base(q_term, d_term, job_no)
            if os.path.exists(save_path):
                mtime = os.path.getmtime(save_path)
                if base_time is None:
                    base_time = mtime
                return int(mtime - base_time)
            else:
                return None

        st = get_m_time(first_job_no)
        ed = get_m_time(last_job_no)
        try:
            elapsed = ed - st
        except TypeError:
            elapsed = None

        print("{} {}".format(i, st / (i+0.01)))


if __name__ == "__main__":
    main()
