import logging
import sys
from typing import Iterable, Tuple

from omegaconf import OmegaConf

from adhoc.bm25_class import BM25
from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path
from trainer_v2.per_project.transparency.mmp.retrieval_run.run_bm25t import load_table_from_conf, to_value_dict


def main():
    conf_path = sys.argv[1]

    c_log.setLevel(logging.DEBUG)
    # run config
    conf = OmegaConf.load(conf_path)
    run_name = conf.run_name

    mapping = load_table_from_conf(conf)
    value_mapping = to_value_dict(mapping, conf.mapping_val)

    # Dataset config
    dataset_conf_path = conf.dataset_conf_path
    dataset_conf = OmegaConf.load(dataset_conf_path)
    dataset_name = dataset_conf.dataset_name
    data_size = dataset_conf.data_size
    quad_tsv_path = dataset_conf.rerank_payload_path

    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    scores_path = get_line_scores_path(run_name, dataset_name)

    # Prediction
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(value_mapping, bm25.core)
    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    predict_qd_itr_save_score_lines(
        bm25t.score,
        qd_iter,
        scores_path,
        data_size
        )

    # Evaluation
    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric)


if __name__ == "__main__":
    main()
