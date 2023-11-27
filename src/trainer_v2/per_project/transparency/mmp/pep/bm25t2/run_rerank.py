import logging
import sys
from omegaconf import OmegaConf
from typing import List, Iterable, Callable, Dict, Tuple, Set
from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import load_bm25_index_resource_conf
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from datastore.sql_based_cache_client import SQLBasedCacheClientS
from misc_lib import select_third_fourth, TimeEstimator, remove_duplicate
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.keras_server.name_short_cuts import get_cached_client
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path
from trainer_v2.per_project.transparency.mmp.pep.bm25t2.bm25t2_scorer import BM25T2
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import get_term_pair_predictor_fixed_context
from trainer_v2.per_project.transparency.mmp.pep.local_decision_helper import load_ts_concat_local_decision_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_scorer_fn(conf) -> Callable[[List[Tuple[str, str]]], List[float]]:
    def pep_fn(items):
        if not items:
            return []
        c_log.warn("This function should not be called")
        c_log.info("%d items", len(items))
        raise ValueError()
        # ret = pep.score_fn(items)
        c_log.info("Done")
        return ret

    def hash_fn(item: Tuple[List[str], List[str]]):
        a, b = item
        return " ".join(a) + "[SEP]" + " ".join(b)

    sqlite_path = conf.cache_path
    cache_client = SQLBasedCacheClientS(
        pep_fn,
        hash_fn,
        0.035,
        sqlite_path)

    pep_fn = cache_client.predict

    c_log.info("Building scorer")
    bm25_conf = load_bm25_index_resource_conf(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    bm25_scoring_fn: Callable[[int, int, int, int], float] = build_bm25_scoring_fn(cdf, avdl)
    bm25t2 = BM25T2(pep_fn, bm25_scoring_fn, df)
    return bm25t2.score


def get_scorer_fn_for_precomputing(conf) -> Callable[[Iterable[Tuple[str, str]]], List[float]]:
    score_term_pair = get_term_pair_predictor_fixed_context(conf.model_path)

    def score_term_pair_wrap(items):
        if not items:
            return []
        c_log.info("%d items", len(items))
        ret = score_term_pair(items)
        c_log.info("Done")
        return ret

    def hash_fn(item: Tuple[List[str], List[str]]):
        a, b = item
        return " ".join(a) + "[SEP]" + " ".join(b)

    sqlite_path = conf.cache_path
    cache_client = SQLBasedCacheClientS(
        score_term_pair_wrap,
        hash_fn,
        0.035,
        sqlite_path)

    todo = []
    def score_term_pair_dummy(payload):
        todo.extend(payload)
        ret = [0 for _ in payload]
        return ret

    def run_computation():
        todo_unique = remove_duplicate(todo, hash_fn)
        c_log.info("%d unique items to predict", len(todo_unique))
        cache_client.predict(todo_unique)

    c_log.info("Building dummy_scorer")
    bm25_conf = load_bm25_index_resource_conf(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    bm25_scoring_fn: Callable[[int, int, int, int], float] = build_bm25_scoring_fn(cdf, avdl)
    bm25t2 = BM25T2(score_term_pair_dummy, bm25_scoring_fn, df)
    return bm25t2.score, run_computation


def main():

    c_log.info(__file__)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    run_name = conf.run_name
    run_precompute = False
    c_log.setLevel(logging.DEBUG)
    # run config
    # Dataset config
    dataset_conf_path = conf.dataset_conf_path
    dataset_conf = OmegaConf.load(dataset_conf_path)
    dataset_name = dataset_conf.dataset_name
    data_size = dataset_conf.data_size
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path

    quad_tsv_path = dataset_conf.rerank_payload_path
    scores_path = get_line_scores_path(run_name, dataset_name)
    f = open(scores_path, "w")
    #
    # # Prediction
    # score_fn: Callable[[List[Tuple[str, str]]], List[float]] = get_scorer_fn(conf)
    if run_precompute:
        c_log.info("Run precomouting")
        score_fn, do_computations = get_scorer_fn_for_precomputing(conf)
    else:
        c_log.info("Use scores to rank")
        score_fn = get_scorer_fn(conf)

    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    ticker = TimeEstimator(data_size)
    for q, d in qd_iter:
        arg = [(q, d)]
        score = score_fn(arg)[0]
        f.write("{}\n".format(score))
        ticker.tick()
    f.close()

    if run_precompute:
        do_computations()

    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path,
        quad_tsv_path, scores_path,
        metric, do_not_report=True)


if __name__ == "__main__":
    main()
