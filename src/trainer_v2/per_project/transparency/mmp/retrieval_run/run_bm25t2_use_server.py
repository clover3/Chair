import logging
import sys

from omegaconf import OmegaConf

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.bm25_retriever_helper import get_tokenize_fn
from adhoc.other.bm25t_retriever import BM25T_Retriever2
from adhoc.test_code.inv_index_test import InvIndexReaderClient
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from models.classic.stopword import load_stopwords
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.retrieval_common import load_table


def get_bm25t_retriever_w_server(conf):
    client = InvIndexReaderClient()
    _ = client.get_postings("book")

    table = load_table(conf)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    tokenize_fn = get_tokenize_fn(bm25_conf)

    stopwords = load_stopwords()
    return BM25T_Retriever2(client.get_postings, df, dl, scoring_fn, tokenize_fn, table, stopwords)


def main():
    c_log.setLevel(logging.INFO)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    retriever = get_bm25t_retriever_w_server(conf)
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
