import json
import sys
from typing import List, Iterable, Dict, Tuple

from list_lib import flatten
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry, scores_to_ranked_list_entries
from trec.types import TrecRankedListEntry


def reciprocal_fusion_score(rank1, rank2, k1, k2) -> float:
    score = 1/(k1+rank1) + 1/(k2+rank2)
    return score


def reciprocal_fusion(l1: List[TrecRankedListEntry], l2: List[TrecRankedListEntry], k1, k2) -> List[Tuple[str, float]]:
    rank_d1 = {e.doc_id: e.rank for e in l1}
    rank_d2 = {e.doc_id: e.rank for e in l2}
    inf = 100000 * 100000

    def get_rank(rank_d, doc_id):
        if doc_id in rank_d:
            return rank_d[doc_id]
        else:
            return inf

    all_doc_ids = set()
    all_doc_ids.update([e.doc_id for e in l1])
    all_doc_ids.update([e.doc_id for e in l2])

    output = []
    for doc_id in all_doc_ids:
        rank1 = get_rank(rank_d1, doc_id)
        rank2 = get_rank(rank_d2, doc_id)
        new_score = reciprocal_fusion_score(rank1, rank2, k1, k2)
        output.append((doc_id, new_score))
    return output


def weighted_sum_fusion(l1: List[TrecRankedListEntry], l2: List[TrecRankedListEntry], k1, k2) -> List[Tuple[str, float]]:
    def weighted_fusion_score(score1, score2, k1, k2) -> float:
        assert score1 <= 1
        assert score2 <= 1
        return k1 * score1 + k2 * score2

    score_d1 = {e.doc_id: e.score for e in l1}
    score_d2 = {e.doc_id: e.score for e in l2}

    def get_score(score_d, doc_id):
        if doc_id in score_d:
            return score_d[doc_id]
        else:
            return 0

    all_doc_ids = set()
    all_doc_ids.update([e.doc_id for e in l1])
    all_doc_ids.update([e.doc_id for e in l2])

    output = []
    for doc_id in all_doc_ids:
        score1 = get_score(score_d1, doc_id)
        score2 = get_score(score_d2, doc_id)
        new_score = weighted_fusion_score(score1, score2, k1, k2)
        output.append((doc_id, new_score))
    return output


def main():
    run_config = json.load(open(sys.argv[1], "r"))

    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(run_config['first_list'])
    l2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(run_config['second_list'])
    run_name = run_config['run_name']
    strategy = run_config['strategy']
    save_path = run_config['save_path']
    k1 = run_config['k1']
    k2 = run_config['k2']
    new_entries: Dict[str, List[TrecRankedListEntry]] = l1

    qid_list = l1.keys()
    for key in l2:
        if key not in qid_list:
            print("WARNING qid {} is not in the first list".format(key))

    for qid in qid_list:
        if qid not in l2:
            new_entries[qid] = l1[qid]
        else:
            entries1 = l1[qid]
            entries2 = l2[qid]
            if strategy == "reciprocal":
                fused_scores = reciprocal_fusion(entries1, entries2, k1, k2)
            elif strategy == "weighted_sum":
                fused_scores = weighted_sum_fusion(entries1, entries2, k1, k2)
            else:
                assert False
            new_entries[qid] = scores_to_ranked_list_entries(fused_scores, run_name, qid)

    flat_entries: Iterable[TrecRankedListEntry] = flatten(new_entries.values())
    write_trec_ranked_list_entry(flat_entries, save_path)


if __name__ == "__main__":
    main()
