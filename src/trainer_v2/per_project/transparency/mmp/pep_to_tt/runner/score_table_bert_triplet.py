import sys

from trainer_v2.per_project.transparency.misc_common import read_term_pair_table
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import predict_with_fixed_context_model_and_save
from cpath import output_path
from misc_lib import path_join


def main():
    job_no = int(sys.argv[1])

    # file_no = job_no // 10
    # sub_no = job_no % 10

    file_no = 0
    model_path = path_join(output_path, "model", "runs", "mmp_pep10_point", "model_20000")
    src_file_path = path_join(output_path, "mmp", "bt2", "align_cands", f"{file_no}.txt")
    log_path = path_join(output_path, "mmp", "bt2", "align_cands_scored", f"{file_no}.txt")

    candidate_pairs = read_term_pair_table(src_file_path)
    num_items = len(candidate_pairs)
    predict_with_fixed_context_model_and_save(model_path, log_path, candidate_pairs, 100, num_items)


if __name__ == "__main__":
    main()
