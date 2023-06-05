import csv
import json
import os
import sys

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from tab_print import tab_print
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path_base
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.summarize_te import get_score
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")
    missing_save = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1_remain.jsonl")

    sub_job_list = get_mmp_split_w_deep_scores()
    todo_list = [line.strip() for line in open(cand_path, "r")]
    f_out = open(missing_save, "w")
    for i in range(0, 10000):
        q_term, d_term = todo_list[i].split()

        def file_exists(job_no):
            save_path = get_te_save_path_base(q_term, d_term, job_no)
            return os.path.exists(save_path)

        sub_job_todo = []
        for j in sub_job_list:
            if not file_exists(j):
                sub_job_todo.append(j)

        if sub_job_todo:
            print("Job {} has {} remain".format(i, len(sub_job_todo)))
            row = [i, sub_job_todo]
            f_out.write(json.dumps(row) + "\n")




if __name__ == "__main__":
    main()
