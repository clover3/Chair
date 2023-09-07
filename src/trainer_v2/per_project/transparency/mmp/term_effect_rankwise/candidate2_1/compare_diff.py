from collections import Counter

from table_lib import tsv_iter
import sys

from misc_lib import print_dict_tab


def main():
    rows = tsv_iter(sys.argv[1])
    count = Counter()
    for row in rows:
        q_term, d_term, score1, score2 = row
        d1 = float(score1) > 0
        d2 = float(score2) > 0

        # if d1 == d2:
        #     pass
        # else:
        #     print("\t".join(row))

        count[f"pearson_{d1}"] += 1
        count[f"spearman_{d2}"] += 1

    print_dict_tab(count)


if __name__ == "__main__":
    main()