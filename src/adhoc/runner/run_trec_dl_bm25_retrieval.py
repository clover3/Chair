import sys

from omegaconf import OmegaConf

from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.bm25_retriever_helper import get_bm25_retriever_from_conf
from adhoc.conf_helper import BM25IndexResource, load_omega_config

from dataclasses import dataclass
from typing import Optional


@dataclass
class BM25RetrievalRunConfigType:
    bm25conf_path: str
    method: str
    run_name: Optional[str]
    dataset_conf_path: str
    outer_batch_size: int = 10000000


def main():
    bm25conf_path = sys.argv[1]
    bm25_conf = load_omega_config(bm25conf_path, BM25IndexResource, True)

    d = {
        'method': "bm25_" + bm25_conf.index_name,
        'dataset_conf_path': "confs/dataset_conf/retrieval_trec_dl_2019.yaml",
        'outer_batch_size': 1000000,
    }
    conf = OmegaConf.create(d)

    retriever = get_bm25_retriever_from_conf(bm25_conf)
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
