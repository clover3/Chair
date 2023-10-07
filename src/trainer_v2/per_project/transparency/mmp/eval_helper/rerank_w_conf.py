import logging
import sys
from typing import Iterable, Tuple

from omegaconf import OmegaConf

from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines, \
    batch_score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path


def run_rerank_with_conf_common(get_scorer_fn):
    conf_path = sys.argv[1]
    c_log.setLevel(logging.DEBUG)
    # run config
    conf = OmegaConf.load(conf_path)
    run_name = conf.run_name
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
    score_fn = get_scorer_fn(conf)
    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))

    try:
        outer_batch_size = conf.outer_batch_size
    except AttributeError:
        outer_batch_size = None

    if outer_batch_size is None:
        predict_qd_itr_save_score_lines(
            score_fn,
            qd_iter,
            scores_path,
            data_size
        )
    else:
        batch_score_and_save_score_lines(
            score_fn,
            qd_iter,
            scores_path,
            data_size,
            outer_batch_size)
    # Evaluation
    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric)