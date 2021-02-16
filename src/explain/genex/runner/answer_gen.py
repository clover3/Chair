import argparse
import os
import sys
from typing import List

import numpy as np

from cache import load_from_pickle
from cpath import output_path
from explain.genex.load import PackedInstance, load_packed
from explain.genex.save_to_file import save_score_to_file, DropStop, save_score_to_file_term_level, RandomConfig
from misc_lib import exist_or_mkdir

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
        save_dir = os.path.join(output_path, "genex", "runs")
        exist_or_mkdir(os.path.join(output_path, "genex"))
        exist_or_mkdir(save_dir)
        save_path = os.path.join(save_dir, save_name)
        data: List[PackedInstance] = load_packed(data_name)

        if method_name == "random":
            config = RandomConfig
            scores: List[np.array] = [np.random.random([512])] * len(data)
        else:
            scores: List[np.array] = load_from_pickle(score_name)

        if "term_" in method_name:
            save_score_to_file_term_level(data, config, save_path, scores)
        else:
            save_score_to_file(data, config, save_path, scores)
    except:
        raise


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    run(args)
