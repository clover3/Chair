import logging
import math
import sys

from omegaconf import OmegaConf

from adhoc.other.bm25_retriever_helper import get_bm25_scorer_from_conf
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common, \
    point_to_batch_scorer
from utils.parallel import parallel_run, parallel_run2


def score_job(args):
    c_log.setLevel(logging.WARN)
    qd_list, bm25_conf = args
    bm25 = get_bm25_scorer_from_conf(bm25_conf)
    return bm25.batch_score(qd_list)


def get_score_fn(conf):
    c_log.info("Building scorer")
    bm25_conf = OmegaConf.load(conf.bm25conf_path)

    def parallel_score_fn(qd_list):
        n_worker = 20
        ret = parallel_run2(qd_list, bm25_conf, score_job, n_worker)
        return ret

    return parallel_score_fn


def main():
    c_log.info(__file__)
    c_log.setLevel(logging.DEBUG)
    get_scorer_fn = get_score_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    run_rerank_with_conf_common(conf, get_scorer_fn, do_not_report=True)


if __name__ == "__main__":
    main()
