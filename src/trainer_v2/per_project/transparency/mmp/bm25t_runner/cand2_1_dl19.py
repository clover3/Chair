from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_qid_pid_scores
from adhoc.eval_helper.pytrec_helper import eval_by_pytrec_json_qrel
from dataset_specific.msmarco.passage.doc_indexing.retriever import load_bm25_resources
from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019, get_rerank_payload_save_path, \
    get_mmp_test_qrel_json_path
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_mapping_from_align_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper
from cpath import output_path
from misc_lib import path_join

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_mapping_from_align_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines
from typing import List, Iterable, Callable, Dict, Tuple, Set



def main():
    ph = get_cand2_1_path_helper()
    table_path = ph.per_pair_candidates.fidelity_table_path
    table_name = f"cand2_1_corr"
    dataset = TREC_DL_2019

    bm25_conf = path_join("confs", "bm25_resource", "sp_stem.yaml")
    cut = 0.1
    mapping_val = 0.1
    method_name = f"bm25_{table_name}"
    metric = "ndcg_cut_10"
    base_run_name = "TREC_DL_2019_BM25_sp_stem"
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)

    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_conf)
    bm25 = BM25(df, cdf, avdl, 0.1, 100, 1.4)
    bm25t = BM25T(mapping, bm25.core)
    quad_tsv_path = get_rerank_payload_save_path(base_run_name)
    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    run_name = f"{method_name}_{dataset}"
    line_scores_path = path_join(output_path, "lines_scores", f"{run_name}.txt")

    # Run predictions and save into lines
    predict_qd_itr_save_score_lines(
        bm25t.score,
        qd_iter,
        line_scores_path,
        200 * 1000)

    # Translate score lines into ranked list
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}.txt")
    build_ranked_list_from_qid_pid_scores(
        quad_tsv_path,
        run_name,
        ranked_list_path,
        line_scores_path)

    # evaluate
    judgment_path = get_mmp_test_qrel_json_path(dataset)
    ret = eval_by_pytrec_json_qrel(
        judgment_path,
        ranked_list_path,
        metric)
    print(ret)


if __name__ == "__main__":
    main()
