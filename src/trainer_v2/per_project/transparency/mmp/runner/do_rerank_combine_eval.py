import sys

from omegaconf import OmegaConf
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
from adhoc.conf_helper import create_omega_config
from adhoc.eval_helper.line_format_to_trec_ranked_list import build_ranked_list_from_line_scores_and_eval
from adhoc.resource.dataset_conf_helper import get_dataset_conf
from adhoc.resource.scorer_loader import get_rerank_scorer, RerankScorerWrap
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import get_line_scores_path
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_u_conf2, \
    RerankDatasetConf, RerankRunConf


# Dataset specific or method specific should be in adhoc.resource
def work(method, dataset):
    rerank_scorer: RerankScorerWrap = get_rerank_scorer(method)
    if not method.startswith("rr_"):
        method = "rr_" + method
    dataset_conf: RerankDatasetConf = get_dataset_conf(dataset)
    outer_batch_size = rerank_scorer.get_outer_batch_size()
    conf: RerankRunConf = create_omega_config(
        {
            "dataset_conf": dataset_conf,
            "method": method,
            "run_name": method,
            "outer_batch_size": outer_batch_size,
        },  RerankRunConf
    )
    if dataset_conf.dataset_name.startswith("dev1K_A_"):
        conf.do_not_report = True
    run_name = conf.run_name
    # Dataset config
    dataset_conf = conf.dataset_conf
    dataset_name = dataset_conf.dataset_name
    quad_tsv_path = dataset_conf.rerank_payload_path
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    do_not_report = False
    scores_path = get_line_scores_path(run_name, dataset_name)
    # Prediction
    build_ranked_list_from_line_scores_and_eval(
        run_name, dataset_name, judgment_path, quad_tsv_path, scores_path,
        metric, do_not_report)


def main():
    c_log.info(__file__)
    method = sys.argv[1]
    try:
        dataset = sys.argv[2]
    except IndexError:
        dataset = "dev_c"

    run_name = f"{method}_{dataset}"
    with JobContext(run_name):
        work(method, dataset)


if __name__ == "__main__":
    main()
