import argparse
import sys
from typing import List

import numpy as np
import scipy.stats as stats

from cache import load_from_pickle

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--data_name", help="data_name")


def main():

    rel_scores: List[np.array] = load_from_pickle(sys.argv[1])

    token_scores: List[np.array] = load_from_pickle(sys.argv[2])

    for rel_score, token_score in zip(rel_scores, token_scores):
        prob = rel_score[1]
        # print(rel_score[1])
        # prob - after_score = token_score
        # prob - token_score = after_score
        after_score = prob - token_score

        explainer_predicted_score_list = []
        for j in range(len(token_score)):
            x_prime = np.ones([len(token_score)])
            x_prime[j] = 0
            explainer_predicted_score = np.dot(token_score, x_prime)
            explainer_predicted_score_list.append(explainer_predicted_score)
        tau, p_value = stats.kendalltau(explainer_predicted_score_list, after_score)
        # print(token_score[:30])
        # print(after_score[:30])
        # print(tau, p_value)
        print(tau, prob)



if __name__ == "__main__":
    main()