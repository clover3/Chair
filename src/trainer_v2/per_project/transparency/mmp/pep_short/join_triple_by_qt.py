import sys

from misc_lib import group_by, get_first
from table_lib import tsv_iter


def main():
    itr = tsv_iter(sys.argv[1])
    out_path = sys.argv[2]
    grouped = group_by(itr, get_first)

    f = open(out_path, "w")
    for qt, items in grouped.items():
        row = [qt]
        for _qt, dt, score in items:
            row.extend([dt, score])
        f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
