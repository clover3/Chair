from cpath import output_path
from misc_lib import path_join
import sys

from table_lib import tsv_iter


def main():

    term_list_path = path_join(output_path, "mmp", "bias", "car_maker_list_under.txt")
    table_path = sys.argv[1]
    table_scores = tsv_iter(table_path)
    score_d = {t: float(s) for _qterm, t, s in table_scores}

    for line in open(term_list_path, "r"):
        term = line.strip()
        print(f"{term}\t{score_d[term]}")



if __name__ == "__main__":
    main()
