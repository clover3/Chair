

# LuK = Lucene tokenizer Krovetz stemmed
import logging
import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from cpath import yconfig_dir_path
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.retrieval_common import get_bm25t_retriever_in_memory


def run_bm25t_luk_trec_dl19(run_name, table_path):
    conf = OmegaConf.create(
        {
            "bm25conf_path": path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml"),
            "dataset_conf_path": path_join(yconfig_dir_path, "dataset_conf", "retrieval_trec_dl_2019_43.yaml"),
            "table_path": table_path,
            "table_type": "Score",
            "method": run_name,
            "run_name": run_name
        }
    )
    retriever = get_bm25t_retriever_in_memory(conf)
    run_retrieval_eval_report_w_conf(conf, retriever)
    c_log.info("Done")


def main():
    c_log.setLevel(logging.INFO)
    table_path = sys.argv[1]
    run_name = sys.argv[2]

    run_bm25t_luk_trec_dl19(run_name, table_path)


if __name__ == "__main__":
    main()
