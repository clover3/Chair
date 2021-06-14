import random
from typing import List, Dict, Tuple, Iterable

from misc_lib import get_second, group_by
from trec.types import TrecRankedListEntry


def score_d_to_trec_style_predictions(score_d: Dict[Tuple[str, str], float], run_name="runname",
                                      max_entry=-1,
                                      shuffle_sort=False) \
        -> Iterable[TrecRankedListEntry]:

    def get_qid(x) -> str:
        return x[0][0]
    qid_grouped: Dict[str, List[Tuple[Tuple[str, str], float]]] = group_by(score_d.items(), get_qid)
    for qid, entries in qid_grouped.items():
        l: List[Tuple[Tuple[str, str], float]] = list(entries)
        if shuffle_sort:
            random.shuffle(l)
        l.sort(key=get_second, reverse=True)
        query_id = str(qid)
        if max_entry > 0:
            l = l[:max_entry]
        for rank, (pair_id, score) in enumerate(l):
            _, candidate_id = pair_id
            doc_id = str(candidate_id)
            yield TrecRankedListEntry(query_id, doc_id, rank, score, run_name)
