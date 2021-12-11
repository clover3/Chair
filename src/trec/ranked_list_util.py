from typing import List

from trec.types import TrecRankedListEntry


def assign_rank(l: List[TrecRankedListEntry]) -> List[TrecRankedListEntry]:
    l.sort(key=lambda x:x.score, reverse=True)
    l_out = []
    for _rank, e in enumerate(l):
        rank = _rank + 1
        e = TrecRankedListEntry(e.query_id, e.doc_id, rank, e.score, e.run_name)
        l_out.append(e)
    return l_out


def remove_duplicates_from_ranked_list(ranked_list_grouped, duplicate_doc_ids):
    new_ranked_list_grouped = {}
    for qid, ranked_list in ranked_list_grouped.items():
        ranked_list_flitered = [e for e in ranked_list if e.doc_id not in duplicate_doc_ids]
        new_ranked_list = assign_rank(ranked_list_flitered)
        new_ranked_list_grouped[qid] = new_ranked_list
    return new_ranked_list_grouped