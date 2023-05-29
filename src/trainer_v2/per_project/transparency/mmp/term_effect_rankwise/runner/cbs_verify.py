import os.path
import sys

from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path
from misc_lib import path_join, ceil_divide
from cpath import output_path



def main():
    job_no_list = []
    dir_path = path_join(output_path, "msmarco", "passage", "tfs_and_scores")

    def get_sub_file_path(sub_job_no):
        return path_join(dir_path, f"{job_no}_{sub_job_no}")

    for job_no in range(0, 100):
        for sub_job_no in range(1000):
            if not os.path.exists(get_sub_file_path(sub_job_no)):
                break

        f = open(get_mmp_grouped_sorted_path(job_no), "r")
        cnt = 0
        for line in f:
            cnt += 1

        n_expected_job = ceil_divide(cnt,  1000 * 100)
        if n_expected_job != sub_job_no:
            print(f"{job_no}: {n_expected_job}: {sub_job_no}")



if __name__ == "__main__":
    main()