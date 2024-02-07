import logging
import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.retrieval_common import get_bm25t_retriever_in_memory


def main():
    c_log.setLevel(logging.INFO)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    with JobContext(f"bm25t_2_{conf.method}"):
        retriever = get_bm25t_retriever_in_memory(conf)
        run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
