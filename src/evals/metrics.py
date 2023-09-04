from typing import List, Callable

from evals.mean_average_precision import get_ap, get_rr
from trec.types import TrecRankedListEntry

MetricFn = Callable[[List[TrecRankedListEntry]], float]


def get_recall_at_k(k) -> MetricFn:
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


def get_ndcg_at_k(k) -> MetricFn:
    from sklearn.metrics import ndcg_score
    def fn(ranked_list: List[TrecRankedListEntry], true_gold: List[str]):
        pred_scores = []
        true_label = []
        for e in ranked_list:
            pred_scores.append(e.score)
            label = 1 if e.doc_id in true_gold else 0
            true_label.append(label)

        return ndcg_score([true_label], [pred_scores], k=k)
    return fn


def get_p_at_k(k) -> MetricFn:
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


def get_is_metric_at_k(metric_prefix) -> MetricFn:
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


def get_metric_fn(input_text) -> MetricFn:
    is_recall_at_k = get_is_metric_at_k("R")
    is_precision_at_k = get_is_metric_at_k("P")

    k = is_recall_at_k(input_text)
    if k:
        return get_recall_at_k(k)
    k = is_precision_at_k(input_text)
    if k:
        return get_p_at_k(k)

    if input_text.lower() == "map":
        return get_ap
    elif input_text.startswith("ndcg"):
        k = int(input_text[len("ndcg"):])
        print("using ndcg at {}".format(k))
        return get_ndcg_at_k(k)
    elif input_text.lower() == "mrr":
        return get_rr


    raise KeyError("Metric {} is not supported".format(input_text))
    assert False