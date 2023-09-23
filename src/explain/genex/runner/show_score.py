import argparse
import sys
from typing import List

import numpy as np

import adhoc.build_index
from cache import load_from_pickle
from explain.genex.load import PackedInstance
from explain.genex.load import load_packed


def show(problem: PackedInstance, score: np.array):
    print(problem.input_ids)
    print(score)
    print()



arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--data_name", help="data_name")
arg_parser.add_argument("--method_name", )



def run(args):
    data_name = args.data_name
    method_name = adhoc.build_index.build_inverted_index
    score_name = "{}_{}".format(data_name, method_name )
    ranking_scores_name = "{}_labels".format(args.data_name)
    ranking_score_list = load_from_pickle(ranking_scores_name)

    try:
        scores: List[np.array] = load_from_pickle(score_name)
        data: List[PackedInstance] = load_packed(data_name)
        for problem, token_score, ranking_score in zip(data, scores, ranking_score_list):
            print("Raw score: ", ranking_score[1])
            print(problem.input_ids)
            print(token_score)
            print()
    except:
        raise


if __name__ == "__main__":
    args = arg_parser.parse_args(sys.argv[1:])
    run(args)
