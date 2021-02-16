import argparse
import os
import sys
from typing import List

import numpy as np

from cache import load_from_pickle
from cpath import output_path
from explain.genex.load import PackedInstance, load_packed
from explain.genex.save_to_file import DropStop, save_score_to_file_term_level

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--data_name", help="data_name")
arg_parser.add_argument("--method_name", )


def run(args):
    data_name = args.data_name
    method_name = args.method_name
    score_name = "{}_{}".format(data_name, method_name)
    config = DropStop
    try:
        save_name = "{}_{}.txt".format(score_name, config.name)
        save_path = os.path.join(output_path, "genex", "runs", save_name)
        scores: List[np.array] = load_from_pickle(score_name)
        data: List[PackedInstance] = load_packed(data_name)
        save_score_to_file_term_level(data, config, save_path, scores)
    except:
        raise


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    run(args)
