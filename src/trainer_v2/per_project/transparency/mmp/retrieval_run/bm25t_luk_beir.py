import logging
import sys
from omegaconf import OmegaConf
from cpath import yconfig_dir_path
from dataset_specific.beir_eval.beir_common import beir_dataset_list_A, beir_mb_dataset_list
from dataset_specific.beir_eval.preprocess.index_corpus2 import build_beir_luk_conf
from dataset_specific.beir_eval.run_helper import run_retrieval_and_eval_on_beir
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.retrieval_common import get_bm25t_in_memory_inner, load_table


def run_bm25t_luk_trec_dl19(method, table_path, dataset_list):
    conf_d = {
        "table_path": table_path,
        "table_type": "Score",
        "method": method,
    }
    conf = OmegaConf.create(conf_d)
    table = load_table(conf)

    for dataset in dataset_list:
        run_name = f"{dataset}_{method}"
        with JobContext(run_name):
            c_log.info("Running for %s", dataset)
            bm25_conf = build_beir_luk_conf(dataset)
            retriever = get_bm25t_in_memory_inner(bm25_conf, table)
            max_doc_per_list = 1000
            method = conf.method
            split = "test"
            run_retrieval_and_eval_on_beir(dataset, split, method, retriever, max_doc_per_list)
            c_log.info("Done")


def main():
    c_log.setLevel(logging.INFO)
    table_path = sys.argv[1]
    method = sys.argv[2]
    if len(sys.argv) > 3:
        dataset_list = [sys.argv[3]]
    else:
        dataset_list = beir_mb_dataset_list
    c_log.info("Todo: %s", str(dataset_list))
    run_bm25t_luk_trec_dl19(method, table_path, dataset_list)


if __name__ == "__main__":
    main()
