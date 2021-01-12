from typing import Dict, Counter

from arg.perspectives.runner_qck.qk_filtering_w_gold_lm import get_query_lms
from arg.qck.filter_qk import filter_qk, filter_qk_rel
from cache import load_from_pickle, save_to_pickle


def get_qk_candidate(split):
    return load_from_pickle("pc_qk2_{}".format(split))


def main1():
    split = "train"
    qk_candidate = get_qk_candidate(split)
    query_lms: Dict[str, Counter] = get_query_lms(split)
    print(len(qk_candidate), len(query_lms))
    filtered_qk_candidate = filter_qk(qk_candidate, query_lms)
    save_to_pickle(filtered_qk_candidate, "pc_qk2_filtered_{}".format(split))


def main_hp09():
    split = "train"
    qk_candidate = get_qk_candidate(split)
    query_lms: Dict[str, Counter] = get_query_lms(split)
    print(len(qk_candidate), len(query_lms))
    alpha = 0.9
    filtered_qk_candidate = filter_qk(qk_candidate, query_lms, alpha)
    save_to_pickle(filtered_qk_candidate, "pc_qk2_09_filtered_{}".format(split))


def main():
    split = "train"
    qk_candidate = get_qk_candidate(split)
    query_lms: Dict[str, Counter] = get_query_lms(split)
    print(len(qk_candidate), len(query_lms))
    filtered_qk_candidate = filter_qk_rel(qk_candidate, query_lms, 50)
    save_to_pickle(filtered_qk_candidate, "pc_qk2_filtered_rel_{}".format(split))


if __name__ == "__main__":
    main_hp09()
