import logging
import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from cpath import yconfig_dir_path
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.retrieval_common import get_bm25t_retriever_in_memory


def main():
    c_log.setLevel(logging.INFO)
    table_path = sys.argv[1]
    run_name = sys.argv[2]
    conf = OmegaConf.create(
        {
            "bm25conf_path": path_join(yconfig_dir_path, "bm25_resource", "bt2.yaml"),
            "dataset_conf_path": path_join(yconfig_dir_path, "dataset_conf", "retrieval_mmp_dev100.yaml"),
            "table_path": table_path,
            "table_type": "Score",
            "method": run_name,
            "run_name": run_name
        }
    )

    job_name = f"{run_name}_benchmark"
    with JobContext(job_name):
        retriever = get_bm25t_retriever_in_memory(conf)
        run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
