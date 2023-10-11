import json
from typing import Dict, List

from pytrec_eval import RelevanceEvaluator

from misc_lib import average
from trainer_v2.chair_logging import c_log
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def eval_by_pytrec(judgment_path, ranked_list_path, metric, n_query_expected=None):
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list_to_dict(ranked_list)
    if judgment_path.endswith(".json"):
        qrels = json.load(open(judgment_path, "r"))
    else:
        qrels = load_qrels_structured(judgment_path)

    return eval_by_pytrec_inner(qrels, doc_scores, metric, n_query_expected)


def eval_by_pytrec_json_qrel(judgment_path, ranked_list_path, metric, n_query_expected=None):
    return eval_by_pytrec(judgment_path, ranked_list_path, metric, n_query_expected)


def eval_by_pytrec_inner(qrels, doc_scores, metric, n_query_expected):
    if n_query_expected is not None:
        if n_query_expected != len(qrels):
            c_log.warning("%d queries are expected but qrels has %d queries", n_query_expected, len(qrels))
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    c_log.debug("Computed scores for %d queries", len(score_per_query))
    scores = [score_per_query[qid][metric] for qid in score_per_query]
    if n_query_expected is not None:
        if n_query_expected != len(qrels):
            c_log.warning("%d queries are expected but result has %d scores", n_query_expected, len(scores))
    return average(scores)


def convert_ranked_list_to_dict(ranked_list: Dict[str, List[TrecRankedListEntry]]):
    out_d = {}
    for qid, entries in ranked_list.items():
        per_q = {}
        for e in entries:
            per_q[e.doc_id] = e.score
        out_d[qid] = per_q
    return out_d