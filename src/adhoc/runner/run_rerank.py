import sys

from omegaconf import OmegaConf
from adhoc.conf_helper import create_omega_config
from adhoc.resource.dataset_conf_helper import get_dataset_conf
from adhoc.resource.scorer_loader import get_rerank_scorer, RerankScorerWrap
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
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
    run_rerank_with_u_conf2(rerank_scorer.score_fn, conf)


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
