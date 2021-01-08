import argparse
import sys

from arg.qck.filter_qk_w_ranked_list import filter_with_ranked_list_path
from cache import save_to_pickle

parser = argparse.ArgumentParser(description='')

parser.add_argument("--qk_name")
parser.add_argument("--ranked_list_path")
parser.add_argument("--threshold")
parser.add_argument("--top_k")
parser.add_argument("--save_name")


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    new_qks = filter_with_ranked_list_path(args.qk_name,
                        args.ranked_list_path,
                        float(args.threshold),
                        int(args.top_k))
    save_to_pickle(new_qks, args.save_name)