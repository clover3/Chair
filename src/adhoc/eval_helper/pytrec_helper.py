import json
from typing import Dict, List, Union

from pytrec_eval import RelevanceEvaluator

from misc_lib import average
from trainer_v2.chair_logging import c_log
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def eval_by_pytrec(judgment_path, ranked_list_path, metric, n_query_expected=None):
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list_to_dict(ranked_list)
    qrels = load_qrels_as_structure_from_any(judgment_path)

    return eval_by_pytrec_inner(qrels, doc_scores, metric, n_query_expected)


def load_qrels_as_structure_from_any(judgment_path) -> Dict[str, Dict[str, int]]:
    if judgment_path.endswith(".json"):
        qrels = json.load(open(judgment_path, "r"))
    else:
        qrels = load_qrels_structured(judgment_path)
    return qrels


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


def convert_ranked_list_to_dict(ranked_list: Union[Dict[str, List[TrecRankedListEntry]], List[TrecRankedListEntry]]):
    if isinstance(ranked_list, dict):
        return convert_dict_ranked_list_to_dict(ranked_list)
    elif isinstance(ranked_list, List):
        return convert_flat_ranked_list_to_dict(ranked_list)


def convert_dict_ranked_list_to_dict(ranked_list):
    out_d = {}
    for qid, entries in ranked_list.items():
        per_q = {}
        for e in entries:
            per_q[e.doc_id] = e.score
        out_d[qid] = per_q
    return out_d


def convert_flat_ranked_list_to_dict(ranked_list: List[TrecRankedListEntry]):
    out_d = {}
    for e in ranked_list:
        if e.query_id not in out_d:
            out_d[e.query_id] = {}
        per_q = out_d[e.query_id]
        per_q[e.doc_id] = e.score
    return out_d
