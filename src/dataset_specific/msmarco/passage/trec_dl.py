import json
from typing import List, Tuple
from typing import List, Iterable, Callable, Dict, Tuple, Set

from pytrec_eval import RelevanceEvaluator

from adhoc.adhoc_retrieval import run_retrieval
from adhoc.bm25_retriever import RetrieverIF
from adhoc.json_run_eval_helper import save_json_qres
from dataset_specific.beir_eval.path_helper import get_json_qres_save_path
from dataset_specific.msmarco.passage.path_helper import load_mmp_test_qrel_json, \
    load_mmp_queries, get_mmp_test_qrel_json_path
from misc_lib import average
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log


def run_mmp_retrieval(dataset, method, retriever: RetrieverIF):
    run_name = f"{dataset}_{method}"
    queries: List[Tuple[str, str]] = load_mmp_queries(dataset)
    c_log.info("%d queries", len(queries))
    max_doc_per_query = 1000
    doc_score_d = run_retrieval(retriever, queries, max_doc_per_query)
    save_json_qres(run_name, doc_score_d)


def run_pytrec_eval(judgment_path, doc_score_path, metric="ndcg_cut_10"):
    qrels = json.load(open(judgment_path, "r"))
    doc_scores = json.load(open(doc_score_path, "r"))
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    scores = [score_per_query[qid][metric] for qid in score_per_query]
    return average(scores)


def eval_mmp_run_and_report(dataset, run_name, metric="ndcg_cut_10"):
    json_qres_save_path = get_json_qres_save_path(run_name)
    doc_score_d = json.load(open(json_qres_save_path, "r"))

    # This has path dependency specific TREC_DL_CORPUS

    qrel_path = get_mmp_test_qrel_json_path(dataset)
    qrels = json.load(open(qrel_path, "r"))
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_score_d)
    per_query_scores = [score_per_query[qid][metric] for qid in score_per_query]
    score = average(per_query_scores)
    print(f"metric:\t{score}")
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, "", metric)


def run_mmp_retrieval_eval_report(dataset, method, retriever: RetrieverIF):
    run_name = f"{dataset}_{method}"
    run_mmp_retrieval(dataset, method, retriever)
    eval_mmp_run_and_report(dataset, run_name)

