import csv

from table_lib import tsv_iter
from cpath import output_path
from misc_lib import path_join


def main():
    compare_save_path = path_join(output_path, "msmarco", "passage", "compare.txt")
    save_path = path_join(output_path, "msmarco", "passage", "fidel_diff.txt")

    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')

    entries = []
    for row in tsv_iter(compare_save_path):
        q_term, d_term, score1, score2 = row

        if float(score1) * float(score2) < 0:
            entries.append((q_term, d_term))
            f_out.writerow((q_term, d_term))


if __name__ == "__main__":
    main()
