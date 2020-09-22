import sys

from arg.perspectives.eval_caches import get_ap_list_from_score_d
from cache import load_from_pickle
from tab_print import print_table


def main():
    score_d = load_from_pickle(sys.argv[1])
    split = "dev"
    ap_list, cids = get_ap_list_from_score_d(score_d, split)
    print_table(zip(cids, ap_list))


if __name__ == "__main__":
    main()
