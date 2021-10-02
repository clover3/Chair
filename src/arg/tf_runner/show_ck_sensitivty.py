import sys

from explain.tf2.deletion_scorer import summarize_deletion_score_batch8, show


def main():
    dir_path = sys.argv[1]
    deletion_per_job = 20
    deletion_offset_list = list(range(20, 301, deletion_per_job))
    summarized_result = summarize_deletion_score_batch8(dir_path, deletion_per_job, deletion_offset_list)
    out_file_name = "ck_contribution.html"
    show(out_file_name, summarized_result)


if __name__ == "__main__":
    main()