import logging
from typing import List, Iterable, Callable, Tuple
from typing import Union

from omegaconf import OmegaConf, DictConfig

from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import batch_score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path

BatchScorer = Callable[[List[Tuple[str, str]]], Iterable[float]]
PointScorer = Callable[[Tuple[str, str]], float]
ScorerSig = Union[BatchScorer, PointScorer]


def run_rerank_with_conf_common(
        conf,
        get_scorer_fn: Callable[[DictConfig], ScorerSig],
        do_not_report=False):
    c_log.setLevel(logging.DEBUG)
    # run config
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
        outer_batch_size = 1

    batch_score_and_save_score_lines(
        score_fn,
        qd_iter,
        scores_path,
        data_size,
        outer_batch_size)
    # Evaluation
    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric, do_not_report)