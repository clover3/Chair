

# LuK = Lucene tokenizer Krovetz stemmed
import logging
import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf, \
    run_retrieval_eval_report_w_conf_inner
from cpath import yconfig_dir_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.helper import get_dataset_conf_path
from trainer_v2.per_project.transparency.mmp.retrieval_run.retrieval_common import get_bm25t_retriever3


def run_bm25t_luk(i):
    run_name = "empty"
    conf = OmegaConf.create(
        {
            "bm25conf_path": path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml"),
            "table_path": "none",
            "table_type": "Score",
            "method": run_name,
            "run_name": run_name
        }
    )
    dataset_conf = OmegaConf.create({
        "dataset_name": f"dev_sample1000_{i}",
        "queries_path": f"data/msmarco/dev_sample1000/queries_{i}.tsv",
        "judgment_path": "data/msmarco/qrels.dev.tsv",
        "metric": "recip_rank",
        "max_doc_per_query": 1000,
    })
    retriever = get_bm25t_retriever3(conf)
    run_retrieval_eval_report_w_conf_inner(retriever, dataset_conf, conf.method, False)
    c_log.info("Done")


def main():
    c_log.setLevel(logging.INFO)
    job_no = int(sys.argv[1])
    run_bm25t_luk(job_no)


if __name__ == "__main__":
    main()
