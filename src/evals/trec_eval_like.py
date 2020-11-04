import sys
from typing import List, Dict

from evals.trec import load_ranked_list_grouped, TrecRankedListEntry
from galagos.parse import load_qrels
from misc_lib import average


def get_recall_at_k(k):
    def fn(ranked_list: List[TrecRankedListEntry], true_gold: List[str]):
        top_k_doc_ids = []
        for e in ranked_list[:k]:
            top_k_doc_ids.append(e.doc_id)

        tp = 0
        for doc_id in true_gold:
            if doc_id in top_k_doc_ids:
                tp += 1

        return tp / len(true_gold) if true_gold else 1
    return fn


def get_p_at_k(k):
    def fn(ranked_list: List[TrecRankedListEntry], true_gold: List[str]):
        seen_doc_id = set()
        tp = 0
        for e in ranked_list[:k]:
            if e.doc_id in seen_doc_id:
                print("WARNING doc id {} is duplicated".format(e.doc_id))

            if e.doc_id in true_gold:
                tp += 1
                seen_doc_id.add(e.doc_id)

        return tp / k
    return fn


def get_is_metric_at_k(metric_prefix):
    def fn(input_text):
        l = len(metric_prefix)
        if input_text[:l] == metric_prefix:
            try:
                k = int(input_text[1:])
            except ValueError:
                return False
            return k
        else:
            return False
    return fn


def get_metric_fn(input_text):
    is_recall_at_k = get_is_metric_at_k("R")
    is_precision_at_k = get_is_metric_at_k("P")
    k = is_recall_at_k(input_text)
    if k:
        return get_recall_at_k(k)
    k = is_precision_at_k(input_text)

    if k:
        return get_p_at_k(k)

    assert False


def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]
    metric = sys.argv[3]

    qrels = load_qrels(judgment_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)

    metric_fn = get_metric_fn(metric)

    score_per_query_list = []
    not_found = 0
    for query_id in ranked_list:
        q_ranked_list = ranked_list[query_id]

        try:
            gold_list = qrels[query_id]
            true_gold = list([doc_id for doc_id, score in gold_list if score > 0])
            score_per_query = metric_fn(q_ranked_list, true_gold)
            score_per_query_list.append(score_per_query)
        except KeyError as e:
            not_found += 1

    if not_found:
        print("{} of {} queires not found".format(not_found, len(ranked_list)))

    score = average(score_per_query_list)
    print("{}\t{}".format(metric, score))


if __name__ == "__main__":
    main()