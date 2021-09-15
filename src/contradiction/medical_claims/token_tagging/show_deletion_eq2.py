import json
import sys

from contradiction.medical_claims.token_tagging.deletion_score_to_html import write_deletion_score_to_html
from explain.tf2.deletion_scorer import summarize_deletion_score_batch8


def raw_logit(logit):
    return logit[1]


def main():
    dir_path = sys.argv[1]
    save_name = sys.argv[2]
    info_path = sys.argv[3]
    info = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 5
    max_offset = num_jobs * deletion_per_job

    deletion_offset_list = list(range(0, max_offset, deletion_per_job))
    summarized_result = summarize_deletion_score_batch8(dir_path, deletion_per_job,
                                                        deletion_offset_list,
                                                        raw_logit,
                                                        )
    out_file_name = "{}.html".format(save_name)
    write_deletion_score_to_html(out_file_name, summarized_result, info)


if __name__ == "__main__":
    main()