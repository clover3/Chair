import json

import scipy.special

from contradiction.medical_claims.token_tagging.deletion_score_to_html import write_deletion_score_to_html
from explain.tf2.deletion_scorer import summarize_deletion_score


def get_contradiction_probability(logit):
    return scipy.special.softmax(logit)[2]


def main():
    dir_path = "C:\\work\\Code\\Chair\\output\\biobert_true_pairs_deletion"
    save_name = "biobert_conflict"
    info_path = "C:\\work\\Code\\Chair\\output\\alamri_tfrecord\\biobert_true_pairs.info"
    info = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 5
    max_offset = num_jobs * deletion_per_job
    deletion_offset_list = list(range(0, max_offset, deletion_per_job))
    summarized_result = summarize_deletion_score(dir_path, deletion_per_job,
                                                 deletion_offset_list,
                                                 get_contradiction_probability,
                                                 )
    out_file_name = "{}.html".format(save_name)
    write_deletion_score_to_html(out_file_name, summarized_result, info)


if __name__ == "__main__":
    main()