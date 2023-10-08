import sys

from omegaconf import OmegaConf

from taskman_client.wrapper3 import report_run3, JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.pep_rerank import get_pep_scorer_from_pointwise
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    c_log.info(__file__)
    get_scorer_fn = get_pep_scorer_from_pointwise
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    print(conf)
    with JobContext(conf.run_name):
        strategy = get_strategy()
        with strategy.scope():
            run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
