import sys
from typing import List, Dict

from scipy.stats import pearsonr

from misc_lib import Averager
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    rlg1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(path1)
    rlg2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(path2)

    pearson_averager = Averager()
    mrr_averager = Averager()
    for key in rlg1:
        rl1 = rlg1[key]
        rl2 = rlg2[key]
        top_doc = rl1[0].doc_id
        doc_rank = list(map(lambda e: e.doc_id, rl2)).index(top_doc)
        rr = 1 / (doc_rank + 1)
        mrr_averager.append(rr)

        rl1.sort(key=lambda e: e.doc_id)
        rl2.sort(key=lambda e: e.doc_id)
        if not len(rl1) == len(rl2):
            print(len(rl1), len(rl2))
            continue
        for e1, e2 in zip(rl1, rl2):
            assert e1.doc_id == e2.doc_id

        scores1 = list(map(lambda x: x.score, rl1))
        scores2 = list(map(lambda x: x.score, rl2))
        r, p = pearsonr(scores1, scores2)
        pearson_averager.append(r)
    print("pearson", pearson_averager.get_average())
    print("mrr", mrr_averager.get_average())


if __name__ == "__main__":
    main()
