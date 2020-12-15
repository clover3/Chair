import os
from typing import List, Dict, Tuple, Iterable

from arg.perspectives.eval_caches import get_eval_candidates_from_pickle
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.types import CPIDPair
from cpath import data_path
from trec.trec_parse import write_trec_relevance_judgement, TrecRankedListEntry, TrecRelevanceJudgementEntry
from list_lib import flatten
from misc_lib import group_by, get_second


def scrore_d_to_trec_style_predictions(score_d: Dict[CPIDPair, float], run_name="runname") -> List[TrecRankedListEntry]:
    cid_grouped: Dict[int, List[Tuple[CPIDPair, float]]] = group_by(score_d.items(), lambda x: x[0][0])
    for cid, entries in cid_grouped.items():
        l: List[Tuple[CPIDPair, float]] = list(entries)
        l.sort(key=get_second, reverse=True)
        query_id = str(cid)
        for rank, (cpid, score) in enumerate(l):
            _, pid = cpid

            doc_id = str(pid)
            yield TrecRankedListEntry(query_id, doc_id, rank, score, run_name)


def get_trec_relevance_judgement() -> Iterable[TrecRelevanceJudgementEntry]:
    gold: Dict[int, List[List[int]]] = get_claim_perspective_id_dict()
    for cid, clusters in gold.items():
        query_id = str(cid)
        pids = set(flatten(clusters))
        for pid in pids:
            e = TrecRelevanceJudgementEntry(query_id, str(pid), 1)
            yield e


def main():
    l = get_trec_relevance_judgement()
    save_path = os.path.join(data_path, "perspective", "qrel.txt")
    write_trec_relevance_judgement(l, save_path)


def save_only_from_candidate():
    l = get_relevance_judgement_only_from_candidate()
    save_path = os.path.join(data_path, "perspective", "qrel_sub.txt")
    write_trec_relevance_judgement(l, save_path)


def get_relevance_judgement_only_from_candidate():
    split = "dev"
    candidates: List[Tuple[int, List[Dict]]] = get_eval_candidates_from_pickle(split)
    valid_set = set()
    for cid, items in candidates:
        for e in items:
            pid = e['pid']
            valid_set.add((cid, pid))
    gold: Dict[int, List[List[int]]] = get_claim_perspective_id_dict()
    l = []
    for cid, clusters in gold.items():
        query_id = str(cid)
        pids = set(flatten(clusters))
        for pid in pids:
            if (cid, pid) in valid_set:
                e = TrecRelevanceJudgementEntry(query_id, str(pid), 1)
                l.append(e)
    return l


if __name__ == "__main__":
    save_only_from_candidate()