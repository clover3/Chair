import logging
from dataclasses import dataclass, field
from typing import List, Iterable, Callable, Tuple, TypedDict, Optional
from typing import Union

from omegaconf import OmegaConf, DictConfig

from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import batch_score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path
from utils.conf_helper import unpack_conf

BatchScorer = Callable[[List[Tuple[str, str]]], Iterable[float]]
PointScorer = Callable[[Tuple[str, str]], float]
ScorerSig = Union[BatchScorer, PointScorer]


def point_to_batch_scorer(point_scorer) -> BatchScorer:
    def batch_score(qd_list):
        output = []
        for q, d in qd_list:
            score = point_scorer(q, d)
            output.append(score)
        return output

    return batch_score


def run_rerank_with_conf_common(
        conf,
        get_scorer_fn: Callable[[DictConfig], ScorerSig],
        do_not_report=False):
    c_log.setLevel(logging.DEBUG)
    # run config
    score_fn = get_scorer_fn(conf)
    c_log.info("run_rerank_with_conf_common")
    run_rerank_with_conf2(score_fn, conf, do_not_report)


def run_rerank_with_conf2(score_fn: ScorerSig, conf, do_not_report=False):
    return run_rerank_with_u_conf2(score_fn, unpack_conf(conf), do_not_report)


@dataclass
class RerankDatasetConf:
    dataset_name: str
    data_size: int
    rerank_payload_path: str
    metric: str
    judgment_path: str
    queries_path: Optional[str]
    src_ranked_list: Optional[str]
    corpus_path: Optional[str]


@dataclass
class RerankRunConf:
    run_name: str
    method: Optional[str]
    dataset_conf: RerankDatasetConf
    outer_batch_size: int
    do_not_report: bool = field(default_factory=lambda: False)


def run_rerank_with_u_conf2(
        score_fn: ScorerSig, conf: RerankRunConf, do_not_report=False):
    run_name = conf.run_name
    # Dataset config
    dataset_conf = conf.dataset_conf
    dataset_name = dataset_conf.dataset_name
    data_size = dataset_conf.data_size
    quad_tsv_path = dataset_conf.rerank_payload_path
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    do_not_report = do_not_report or conf.do_not_report
    scores_path = get_line_scores_path(run_name, dataset_name)
    # Prediction
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


def run_build_ranked_list_from_line_scores_and_eval(conf):
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
    # Evaluation
    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric)

