import sys
from collections import Counter
from typing import List, Dict, Tuple

from arg.pf_common.base import ScoreParagraph
from cache import load_from_pickle, save_to_pickle
from list_lib import lmap
from misc_lib import TimeEstimator
from models.classic.stopword import load_stopwords
from tlm.retrieve_lm.stem import CacheStemmer


def count_it(data: Dict[str, List[ScoreParagraph]]) -> List[Tuple[str, Counter]]:
    stemmer = CacheStemmer()
    r = []
    stopword = load_stopwords()

    def remove_stopwords(tokens: List[str]) -> List[str]:
        return list([t for t in tokens if t not in stopword])

    ticker = TimeEstimator(len(data))
    for cid, para_list in data.items():
        ticker.tick()
        tokens_list: List[List[str]] = [e.paragraph.tokens for e in para_list]
        list_tokens: List[List[str]] = lmap(stemmer.stem_list, tokens_list)
        list_tokens: List[List[str]] = lmap(remove_stopwords, list_tokens)

        all_cnt = Counter()
        for tokens in list_tokens:
            all_cnt.update(Counter(tokens))

        r.append((cid, all_cnt))
    return r


# input : pc_dev_paras_top_100 / pc_train_paas_by_cid
# output :
def main(input_pickle_name, output_pickle_name):
    r = count_it(load_from_pickle(input_pickle_name))
    save_to_pickle(r, output_pickle_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

