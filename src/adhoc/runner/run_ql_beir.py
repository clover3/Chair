import logging
import sys

from omegaconf import OmegaConf

from adhoc.other.ql_retriever_helper import get_ql_retriever_from_conf
from cpath import yconfig_dir_path
from dataset_specific.beir_eval.beir_common import beir_mb_dataset_list
from dataset_specific.beir_eval.preprocess.index_corpus2 import build_beir_luk_conf
from dataset_specific.beir_eval.run_helper import run_retrieval_and_eval_on_beir
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log


def run_ql_luk(run_name, dataset_list):
    conf_d = {
        "ql_conf_path": path_join(
            yconfig_dir_path, "bm25_resource", "lucene_krovetz_ql.yaml"),
        # "table_path": table_path,
        # "table_type": "Score",
        "method": run_name,
        "run_name": run_name
    }
    conf = OmegaConf.create(conf_d)
    ql_conf = OmegaConf.load(conf.ql_conf_path)

    for dataset in dataset_list:
        with JobContext(run_name):
            c_log.info("Running for %s", dataset)
            resource_conf = build_beir_luk_conf(dataset)
            ql_conf.merge_with(resource_conf)
            retriever = get_ql_retriever_from_conf(ql_conf)
            max_doc_per_list = 1000
            method = conf.method
            split = "test"
            run_name = f"{dataset}_{method}"
            run_retrieval_and_eval_on_beir(dataset, split, method, retriever, max_doc_per_list)
            c_log.info("Done")


def main():
    c_log.setLevel(logging.INFO)
    run_name = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_list = [sys.argv[2]]
    else:
        dataset_list = beir_mb_dataset_list
    c_log.info("Todo: %s", str(dataset_list))
    run_ql_luk(run_name, dataset_list)


if __name__ == "__main__":
    main()
