import sys

from omegaconf import OmegaConf

from adhoc.other.retriever_helper import get_bm25_retriever_from_conf
from adhoc.other.retriever_run_w_conf import run_retrieval_from_conf


def main():
    # Run BM25 retrieval
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    # bm25conf_path
    # run_name
    # dataset_conf_path
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    retriever = get_bm25_retriever_from_conf(bm25_conf)
    run_retrieval_from_conf(conf, retriever)


if __name__ == "__main__":
    main()