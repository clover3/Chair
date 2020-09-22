import csv
import sys

from scipy.stats import ttest_rel

from misc_lib import average


def read_csv_as_list(file_path):
    reader = csv.reader(open(file_path, "r"), delimiter="\t")
    for row in reader:
        yield row


def main():
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    a_list = list(read_csv_as_list(file1_path))
    b_dict = dict(read_csv_as_list(file2_path))

    rows = []
    for row in a_list:
        cid, score = row
        a_score = float(score)
        b_score = float(b_dict[cid])
        new_row = (cid, a_score, b_score)
        rows.append(new_row)

    a_series = [r[1] for r in rows]
    b_series = [r[2] for r in rows]

    assert len(a_series) == len(b_series)
    print("{} items".format(len(a_series)))
    print(average(a_series))
    print(average(b_series))
    print(ttest_rel(a_series, b_series))


if __name__ == "__main__":
    main()