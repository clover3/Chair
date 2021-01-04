from typing import Dict, Counter

from arg.perspectives.runner_qck.qk_filtering_w_gold_lm import get_query_lms
from arg.qck.filter_qk import filter_qk
from cache import load_from_pickle, save_to_pickle


def get_qk_candidate(split):
    return load_from_pickle("pc_qk2_{}".format(split))


def main():
    split = "train"
    qk_candidate = get_qk_candidate(split)
    query_lms: Dict[str, Counter] = get_query_lms(split)
    print(len(qk_candidate), len(query_lms))
    filtered_qk_candidate = filter_qk(qk_candidate, query_lms)
    save_to_pickle(filtered_qk_candidate, "pc_qk2_filtered_{}".format(split))


if __name__ == "__main__":
    main()
