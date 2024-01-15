import tensorflow as tf
import sys

from omegaconf import OmegaConf

from adhoc.bm25 import BM25_verbose
from adhoc.bm25_retriever import BM25Retriever, build_bm25_scoring_fn
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.bm25_retriever_helper import get_bm25_retriever_from_conf, get_tokenize_fn
import os

from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf, load_bm25_resources



def build_bm25_scoring_fn(cdf, avdl):
    b = 0.75
    k1 = 1.2
    k2 = 100

    def scoring_fn(tf, qf, dl, qdf):
        return BM25_verbose(tf, qf, qdf, cdf, dl, avdl, b, k1, k2)

    return scoring_fn


def get_bm25_retriever_from_conf(conf, avdl=None, stopwords=None) -> BM25Retriever:
    avdl, cdf, df, dl, inv_index = load_bm25_resources(conf, avdl)
    tokenize_fn = get_tokenize_fn(conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    return BM25Retriever(tokenize_fn, inv_index, df, dl, scoring_fn, stopwords)


def main():
    # Run BM25 retrieval
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    conf.method = "bt2_bm25_param_default"
    # bm25conf_path
    # run_name
    # dataset_conf_path
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    retriever = get_bm25_retriever_from_conf(bm25_conf)
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
