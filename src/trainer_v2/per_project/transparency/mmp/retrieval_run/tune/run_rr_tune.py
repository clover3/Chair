import itertools
from collections import defaultdict

from pytrec_eval import RelevanceEvaluator
from typing import Iterable, Dict, Tuple, List

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.line_format_to_trec_ranked_list import build_rankd_list_from_qid_pid_scores_inner
from adhoc.eval_helper.pytrec_helper import load_qrels_as_structure_from_any, convert_flat_ranked_list_to_dict
from adhoc.other.bm25_retriever_helper import get_tokenize_fn, \
    get_bm25_stats_from_conf
from list_lib import lfrange
from tab_print import print_table
from table_lib import tsv_iter
from cpath import yconfig_dir_path
from misc_lib import path_join, average, select_third_fourth, select_first_second
from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t.bm25t_4 import BM25T_4
from adhoc.resource.dataset_conf_helper import get_rerank_dataset_conf_path


def run_rerank_tuning(param_itr, score_fn_factory):
    dataset_conf_path = get_rerank_dataset_conf_path("dev_C")
    dataset_conf = OmegaConf.load(dataset_conf_path)
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    quad_tsv_path = dataset_conf.rerank_payload_path

    qd_list: List[Tuple[str, str]] = list(select_third_fourth(tsv_iter(quad_tsv_path)))
    qid_pid_list: List[Tuple[str, str]] = list(select_first_second(tsv_iter(quad_tsv_path)))
    qrels: Dict[str, Dict[str, int]] = load_qrels_as_structure_from_any(judgment_path)

    def do_retrieval(score_fn):
        run_name = "tune"
        scores: Iterable[float] = score_fn(qd_list)
        # scores = parallel_run(qd_list, score_fn, 10)
        all_entries = build_rankd_list_from_qid_pid_scores_inner(qid_pid_list, run_name, scores)
        doc_score_d = convert_flat_ranked_list_to_dict(all_entries)
        evaluator = RelevanceEvaluator(qrels, {metric})
        score_per_query = evaluator.evaluate(doc_score_d)
        per_query_scores = [score_per_query[qid][metric] for qid in score_per_query]
        score = average(per_query_scores)
        return score

    c_log.info("Start tuning runs")
    table = []
    for param_cand in param_itr:
        score_fn = score_fn_factory(param_cand)
        score = do_retrieval(score_fn)
        row = (score, param_cand)
        print(row)
        table.append(row)
        
    return table


def get_bm25t_score_fn_factory():
    bm25conf_path = path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml")
    bm25_conf = OmegaConf.load(bm25conf_path)
    value_mapping: Dict[str, Dict[str, float]] = defaultdict(dict)
    tokenize_fn = get_tokenize_fn(bm25_conf)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf, None)
    score_fn_factory = get_score_fn_factory(avdl, cdf, df, tokenize_fn, value_mapping)
    return score_fn_factory


def get_score_fn_factory(avdl, cdf, df, tokenize_fn, value_mapping):
    def score_fn_factory(param_cand):
        b, k1, k2 = param_cand
        scoring_fn = build_bm25_scoring_fn(cdf, avdl, b, k1, k2)
        bm25t = BM25T_4(value_mapping, scoring_fn, tokenize_fn, df)
        score_fn = bm25t.score_batch
        return score_fn
    return score_fn_factory


def main():
    score_fn_factory = get_bm25t_score_fn_factory()
    # 0.20127794059882295	(0.5, 0.8, 100)
    # Best: 0.20804980380568	(0.3, 0.5, 100)

    b_range = lfrange(0.1, 0.4, 0.1)
    k1_range = lfrange(0.1, 0.5, 0.1)
    k2_range = [100]
    itr = itertools.product(b_range, k1_range, k2_range)

    table = run_rerank_tuning(itr, score_fn_factory)
    print_table(table)


if __name__ == "__main__":
    main()
