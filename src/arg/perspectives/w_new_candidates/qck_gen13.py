import os

from arg.perspectives.w_new_candidates.common import qck_gen_w_ranked_list_multiple
from cpath import output_path


def main():
    rough_num_qk = 400
    rough_num_thread = 10
    n_qk_per_job = int(rough_num_qk / rough_num_thread)
    split = "train"
    job_name = "qck13"
    qk_candidate_name = "pc_qk2_filtered_" + split
    ranked_list_path = os.path.join(output_path,
                                    "perspective_experiments",
                                    "pc_qres", "{}.txt".format(split))
    qck_gen_w_ranked_list_multiple(job_name, qk_candidate_name, ranked_list_path, split, n_qk_per_job)


if __name__ == "__main__":
    main()