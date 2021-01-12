import os

from arg.perspectives.w_new_candidates.common import qck_gen_w_ranked_list
from cpath import output_path


def main():
    split = "train"
    job_name = "qck5"
    qk_candidate_name = "pc_qk2_filtered_" + split
    ranked_list_path = os.path.join(output_path,
                                    "perspective_experiments",
                                    "pc_qres", "{}.txt".format(split))
    qck_gen_w_ranked_list(job_name, qk_candidate_name, ranked_list_path, split)


if __name__ == "__main__":
    main()