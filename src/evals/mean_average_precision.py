from typing import List, Dict

from list_lib import lmap
from misc_lib import average
from trec.types import QRelsFlat, TrecRankedListEntry


def get_ap(ranked_list: List[TrecRankedListEntry], true_gold: List[str]):
    num_pred = 0
    n_tp = 0
    prec_list = []
    for e in ranked_list:
        num_pred += 1
        if e.doc_id in true_gold:
            n_tp += 1

            prec = n_tp / num_pred
            prec_list.append(prec)

    while len(prec_list) < len(true_gold):
        prec_list.append(0)

    return average(prec_list)


def get_map(ranked_list_dict: Dict[str, List[TrecRankedListEntry]], qrels: QRelsFlat) -> float:
    def get_ap_wrap(query_id) -> float:
        ranked_list: List[TrecRankedListEntry] = ranked_list_dict[query_id]
        if query_id in qrels:
            judged_entries = qrels[query_id]
            gold_doc_ids = list([doc_id for doc_id, score in judged_entries if score > 0])
            ap = get_ap(ranked_list, gold_doc_ids)
            return ap
        else:
            return 1

    return average(lmap(get_ap_wrap, ranked_list_dict.keys()))


# Assumes ranked list is sorted
def get_rr(ranked_list: List[TrecRankedListEntry], true_gold: List[str]):
    rank_at_correct = -1
    for rank_0, e in enumerate(ranked_list):
        if e.doc_id in true_gold:
            rank_at_correct = rank_0 + 1
            break

    if rank_at_correct == -1:
        return 0
    else:
        return 1 / rank_at_correct
