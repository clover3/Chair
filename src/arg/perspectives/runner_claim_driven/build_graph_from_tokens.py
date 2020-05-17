from typing import List, Dict, Counter, Tuple

from arg.perspectives.random_walk.count_coocurrence_graph import build_co_occurrence
from cache import load_from_pickle, save_to_pickle
from misc_lib import TimeEstimator
from tlm.retrieve_lm.stem import CacheStemmer


def build_co_occur_from_pc_feature(data: Dict[str, List[List[str]]]) \
        -> List[Tuple[str, Counter]]:
    window_size = 10
    stemmer = CacheStemmer()
    r = []
    ticker = TimeEstimator(len(data))
    for cid, tokens_list in data.items():
        ticker.tick()
        counter = build_co_occurrence(tokens_list, window_size, stemmer)
        r.append((cid, counter))
    return r


def main():
    r = build_co_occur_from_pc_feature(load_from_pickle("dev_claim_docs"))
    save_to_pickle(r, "dev_claim_graph")
    r = build_co_occur_from_pc_feature(load_from_pickle("train_claim_docs"))
    save_to_pickle(r, "train_claim_graph")


if __name__ == "__main__":
    main()

