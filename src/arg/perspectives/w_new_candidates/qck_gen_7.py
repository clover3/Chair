import os

from arg.perspectives.w_new_candidates.qck_gen import qck_gen_w_ranked_list
from cpath import output_path


def main():
    for split in ["dev"]:
        job_name = "qck7"
        qk_candidate_name = "perspective_qk_candidate_{}_filtered".format(split)
        ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        "pc_qres", "{}.txt".format(split))
        qck_gen_w_ranked_list(job_name, qk_candidate_name, ranked_list_path, split)


if __name__ == "__main__":
    main()