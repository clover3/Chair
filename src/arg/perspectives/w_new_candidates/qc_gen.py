import os

from arg.perspectives.load import splits
from arg.perspectives.qck.qck_common import get_qck_queries, get_qck_candidate_from_ranked_list_path
from arg.qck.qc.qc_common import make_pc_qc
from cpath import output_path
from misc_lib import exist_or_mkdir


def main():
    save_dir = os.path.join(output_path, "pc_qc2")
    exist_or_mkdir(save_dir)
    for split in splits:
        queries = get_qck_queries(split)
        q_res_path = os.path.join("output",
                                  "perspective_experiments",
                                  "q_res_{}.txt".format(split))
        eval_candidate = get_qck_candidate_from_ranked_list_path(q_res_path)
        save_path = os.path.join(save_dir, split)
        make_pc_qc(queries, eval_candidate, save_path)


if __name__ == "__main__":
    main()