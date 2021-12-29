import sys
from typing import List, Dict

from misc_lib import NamedAverager
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    rlg1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(path1)
    rlg2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(path2)

    t = 0.5

    navg = NamedAverager()
    for key in rlg1:
        rl1 = rlg1[key]
        rl2 = rlg2[key]
        def get_pos_ids(rl: List[TrecRankedListEntry]):
            return [e.doc_id for e in rl if e.score > t]

        pred1 = get_pos_ids(rl1)
        pred2 = get_pos_ids(rl2)
        common = set(pred1).intersection(pred2)


        n_gold = len(pred1)
        n_pred = len(pred2)
        n_tp = len(common)

        precision = n_tp / n_pred if n_pred > 0 else 1
        recall = n_tp / n_gold if n_gold > 0 else 1
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        print(n_gold, n_pred, n_tp)
        navg['precision'].append(precision)
        navg['recall'].append(recall)
        navg['f1'].append(f1)

        rl1.sort(key=lambda e: e.doc_id)
        rl2.sort(key=lambda e: e.doc_id)
        if not len(rl1) == len(rl2):
            print(len(rl1), len(rl2))
            continue
        for e1, e2 in zip(rl1, rl2):
            assert e1.doc_id == e2.doc_id

    for k, v in navg.get_average_dict().items():
        print(k, v)


if __name__ == "__main__":
    main()
