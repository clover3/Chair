from typing import List, Dict, Counter, Tuple

from arg.perspectives.random_walk.count_coocurrence_graph import build_co_occurrence
from arg.pf_common.base import ScoreParagraph
from cache import load_from_pickle, save_to_pickle
from misc_lib import TimeEstimator
from tlm.retrieve_lm.stem import CacheStemmer


def build_co_occur_from_pc_feature(data: Dict[str, List[ScoreParagraph]]) -> List[Tuple[str, Counter]]:
    window_size = 10
    stemmer = CacheStemmer()
    r = []

    ticker = TimeEstimator(len(data))
    for cid, para_list in data.items():
        ticker.tick()
        tokens_list: List[List[str]] = [e.paragraph.tokens for e in para_list]
        counter = build_co_occurrence(tokens_list, window_size, stemmer)
        r.append((cid, counter))
    return r


def main():
    r = build_co_occur_from_pc_feature(load_from_pickle("pc_dev_paras_top_100"))
    save_to_pickle(r, "pc_dev_co_occur_100")


if __name__ == "__main__":
    main()

