import json

from contradiction.medical_claims.token_tagging.show_deletion_score_with_normalization import \
    write_deletion_score_to_html, get_neutral_probability
from explain.tf2.deletion_scorer import summarize_deletion_score_batch8


def main():
    dir_path = "C:\\work\\Code\\Chair\\output\\biobert_alamri1_deletion"
    save_name = "biobert_alamri1_deletion"
    info_path = "C:\\work\\Code\\Chair\\output\\alamri_annotation1\\tfrecord\\biobert_alamri1.info"
    info = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 5
    max_offset = num_jobs * deletion_per_job
    deletion_offset_list = list(range(0, max_offset, deletion_per_job))
    summarized_result = summarize_deletion_score_batch8(dir_path, deletion_per_job,
                                                        deletion_offset_list,
                                                        get_neutral_probability,
                                                        )
    out_file_name = "{}.html".format(save_name)
    write_deletion_score_to_html(out_file_name, summarized_result, info)


if __name__ == "__main__":
    main()