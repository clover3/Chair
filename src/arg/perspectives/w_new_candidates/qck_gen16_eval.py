import os

from arg.perspectives.w_new_candidates.qck_gen16 import qck_gen
from cpath import output_path


def main():
    for split in ["dev", "test"]:
        job_name = "qck16"
        qk_candidate_name = "pc_qk2_" + split
        query_path = os.path.join(output_path, "perspective_experiments",
                                  "claim_query", "perspective_claim_query2_{}.json".format(split))
        candidate_ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        "pc_qres", "{}.txt".format(split))

        kdp_ranked_list_path = os.path.join(output_path,
                                            "perspective_experiments",
                                            "clueweb_qres", "{}.txt".format(split))
        qck_gen(job_name, qk_candidate_name, query_path,
                candidate_ranked_list_path, kdp_ranked_list_path, split)


if __name__ == "__main__":
    main()
