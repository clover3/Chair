import os

from arg.perspectives.w_new_candidates.common import multi_qck_gen_w_neg
from cpath import output_path


def main():
    for split in ["dev", "test"]:
        job_name = "qck_multi2"
        qk_candidate_name = "pc_qk2_{}".format(split)
        ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        "pc_qres", "{}.txt".format(split))
        k_group_size = 32
        multi_qck_gen_w_neg(job_name, qk_candidate_name, ranked_list_path, split, k_group_size)


if __name__ == "__main__":
    main()