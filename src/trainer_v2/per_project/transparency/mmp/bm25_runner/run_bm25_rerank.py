import logging
import sys

from omegaconf import OmegaConf

from adhoc.other.retriever_helper import get_bm25_scorer_from_conf
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_score_fn(conf):
    c_log.info("Building scorer")
    bm25 = get_bm25_scorer_from_conf(conf)
    return bm25.score


def main():
    c_log.info(__file__)
    c_log.setLevel(logging.DEBUG)
    get_scorer_fn = get_score_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    strategy = get_strategy()
    with strategy.scope():
        run_rerank_with_conf_common(conf, get_scorer_fn, do_not_report=True)


if __name__ == "__main__":
    main()
