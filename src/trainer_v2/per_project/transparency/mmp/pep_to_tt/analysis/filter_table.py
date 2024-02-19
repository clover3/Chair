import csv
import os
import sys

from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.pep_to_tt.analysis.filter_fns import get_filter


def main():
    input_file = sys.argv[1]
    pattern_name = sys.argv[2]

    file_name, file_extension = os.path.splitext(input_file)
    save_path = f"{file_name}_{pattern_name}{file_extension}"

    if os.path.exists(save_path):
        print(f"File {save_path} exists. Termiante")
        return

    tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
    pattern_match = get_filter(pattern_name)
    for q_term, d_term, score in tsv_iter(sys.argv[1]):
        if pattern_match(q_term, d_term, score):
            tsv_writer.writerow([q_term, d_term, score])


if __name__ == "__main__":
    main()
