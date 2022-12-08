from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from list_lib import index_by_fn, all_equal
from misc_lib import get_second
from trainer.promise import PromiseKeeper
from trec.types import TrecRankedListEntry


def assign_rank(l: List[TrecRankedListEntry]) -> List[TrecRankedListEntry]:
    l.sort(key=lambda x:x.score, reverse=True)
    l_out = []
    for _rank, e in enumerate(l):
        rank = _rank + 1
        e = TrecRankedListEntry(e.query_id, e.doc_id, rank, e.score, e.run_name)
        l_out.append(e)
    return l_out


def build_ranked_list(qid, run_name, scored_docs: List[Tuple[str, float]]) -> List[TrecRankedListEntry]:
    scored_docs.sort(key=get_second, reverse=True)
    l_out = []
    for _rank, (doc_id, score) in enumerate(scored_docs):
        rank = _rank + 1
        e = TrecRankedListEntry(qid, doc_id, rank, score, run_name)
        l_out.append(e)
    return l_out


def remove_duplicates_from_ranked_list(ranked_list_grouped, duplicate_doc_ids):
    new_ranked_list_grouped = {}
    for qid, ranked_list in ranked_list_grouped.items():
        ranked_list_flitered = [e for e in ranked_list if e.doc_id not in duplicate_doc_ids]
        new_ranked_list = assign_rank(ranked_list_flitered)
        new_ranked_list_grouped[qid] = new_ranked_list
    return new_ranked_list_grouped


def ensemble_ranked_list(rlg_list: List[Dict[str, List[TrecRankedListEntry]]],
                         combine_fn: Callable[[List[List[float]]], List[float]],
                         run_name
                         )\
        -> Dict[str, List[TrecRankedListEntry]]:

    qid_list = list(rlg_list[0].keys())
    d_out = {}
    for qid in qid_list:
        rl_list = [rlg[qid] for rlg in rlg_list]

        if not all_equal(list(map(len, rl_list))):
            raise ValueError

        # Dict[DocID, List[Entry]]
        e_dict_list: List[Dict[str, TrecRankedListEntry]] = [index_by_fn(lambda x: x.doc_id, rl) for rl in rl_list]
        doc_id_list = list(e_dict_list[0].keys())
        scores_list = []
        for e_dict in e_dict_list:
            scores = [e_dict[doc_id].score for doc_id in doc_id_list]
            scores_list.append(scores)

        new_scores = combine_fn(scores_list)

        e_list = []
        for doc_id, score in zip(doc_id_list, new_scores):
            e = TrecRankedListEntry(qid, doc_id, 0, score, run_name)
            e_list.append(e)

        d_out[qid] = assign_rank(e_list)
    return d_out


def batch_rerank(
        queries: List[Tuple[str, str]],
        docs_d: Dict[str, str],
        rlg: Dict[str, List[TrecRankedListEntry]],
        scorer: Callable[[List[Tuple[str, str]]], List[float]],
        run_name: str
) -> List[TrecRankedListEntry]:
    pk = PromiseKeeper(scorer)
    qid_doc_id_future_list = []
    for qid, query_text in queries:
        for e in rlg[qid]:
            doc_text = docs_d[e.doc_id]
            score_future = pk.get_future((query_text, doc_text))
            e = qid, e.doc_id, score_future
            qid_doc_id_future_list.append(e)
    pk.do_duty(True)
    per_qid: Dict[str, List] = defaultdict(list)
    for qid, doc_id, score_future in qid_doc_id_future_list:
        s: float = score_future.get()
        per_qid[qid].append((doc_id, s))

    tr_entries: List[TrecRankedListEntry] = []
    for qid, scored_docs in per_qid.items():
        rl = build_ranked_list(qid, run_name, scored_docs)
        tr_entries.extend(rl)
    return tr_entries



