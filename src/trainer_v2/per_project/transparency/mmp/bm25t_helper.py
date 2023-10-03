from collections import defaultdict
from typing import Dict

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.path_helper import get_rerank_payload_save_path
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import predict_and_save_scores, \
    eval_dev_mrr, \
    predict_and_save_scores_w_itr, eval_from_score_lines_inner
from typing import List, Iterable, Dict, Tuple


def load_mapping_from_align_scores(
        tsv_path, cut, mapping_val) -> Dict[str, Dict[str, float]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(dict)
    for q_term, d_term, score in rows:
        if float(score) > cut:
            mapping[q_term][d_term] = mapping_val
            n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_mapping_from_align_candidate(
        tsv_path, mapping_val) -> Dict[str, Dict[str, float]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(dict)
    for q_term, d_term in rows:
        mapping[q_term][d_term] = mapping_val
        n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_binary_mapping_from_align_candidate(tsv_path) -> Dict[str, List[str]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(list)
    for q_term, d_term in rows:
        mapping[q_term].append(d_term)
        n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def load_binary_mapping_from_align_scores(
        tsv_path, cut) -> Dict[str, List[str]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(list)
    for row in rows:
        if len(row) == 3:
            q_term, d_term, score = row
            if float(score) > cut:
                mapping[q_term].append(d_term)
                n_entry += 1
        else:
            q_term, d_term = row
            mapping[q_term].append(d_term)
            n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def run_dev_rerank_eval_with_bm25t(dataset, mapping, run_name):
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(mapping, bm25.core)
    predict_and_save_scores(bm25t.score, dataset, run_name, 1000 * 1000)
    score = eval_dev_mrr(dataset, run_name)
    print(f"mrr:\t{score}")
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, dataset, "mrr")
    print(f"Recip_rank:\t{score}")



