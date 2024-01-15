from typing import Iterable, Tuple

from adhoc.bm25_class import BM25
from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_qid_pid_scores
from adhoc.eval_helper.pytrec_helper import eval_by_pytrec_json_qrel
from cpath import output_path
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import load_omega_config_with_dataclass, \
    BM25IndexResource
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from dataset_specific.msmarco.passage.path_helper import get_rerank_payload_save_path, get_mmp_test_qrel_json_path
from misc_lib import path_join, select_third_fourth
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.table_readers import load_mapping_from_align_scores, \
    load_mapping_from_align_candidate
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines


def bm25t_rerank_run_and_eval_from_scores(dataset, table_name, table_path):
    cut = 0.1
    mapping_val = 0.1
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)
    bm25t_nltk_stem_rerank_run_and_eval(dataset, table_name, mapping)


def bm25t_rerank_run_and_eval_from_list(dataset, table_name, table_path):
    mapping_val = 0.1
    mapping = load_mapping_from_align_candidate(table_path, mapping_val)
    bm25t_nltk_stem_rerank_run_and_eval(dataset, table_name, mapping)


def bm25t_nltk_stem_rerank_run_and_eval(dataset, table_name, mapping):
    run_name = f"bm25_{table_name}"
    metric = "ndcg_cut_10"
    base_run_name = "TREC_DL_2019_BM25_sp_stem"
    bm25_conf_path = path_join("confs", "bm25_resource", "stem.yaml")
    bm25_conf = load_omega_config_with_dataclass(bm25_conf_path, BM25IndexResource)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    bm25 = BM25(df, cdf, avdl, 0.1, 100, 1.4)
    bm25t = BM25T(mapping, bm25.core)
    score_fn = bm25t.score
    rerank_mmp(base_run_name, dataset, metric, run_name, score_fn)


def rerank_mmp(base_run_name, dataset, metric, run_name, score_fn):
    quad_tsv_path = get_rerank_payload_save_path(base_run_name)
    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    run_dataset_name = f"{run_name}_{dataset}"
    line_scores_path = path_join(output_path, "lines_scores", f"{run_dataset_name}.txt")
    # Run predictions and save into lines
    predict_qd_itr_save_score_lines(
        score_fn,
        qd_iter,
        line_scores_path,
        200 * 1000)
    # Translate score lines into ranked list
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_dataset_name}.txt")
    build_ranked_list_from_qid_pid_scores(
        quad_tsv_path,
        run_dataset_name,
        ranked_list_path,
        line_scores_path)
    # evaluate
    judgment_path = get_mmp_test_qrel_json_path(dataset)
    ret = eval_by_pytrec_json_qrel(
        judgment_path,
        ranked_list_path,
        metric)
    print("{}\t{}".format(metric, ret))