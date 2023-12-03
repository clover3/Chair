import logging
import sys

from omegaconf import OmegaConf

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.retreival_exp_helper import run_retrieval_eval_report_w_conf
from adhoc.other.bm25_retriever_helper import get_tokenize_fn
from adhoc.other.bm25t_retriever import BM25T_Retriever2
from adhoc.test_code.inv_index_test import InvIndexReaderClient
from dataset_specific.msmarco.passage.doc_indexing.retriever import load_bm25_resources, get_bm25_stats_from_conf
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_align_scores


def load_table(conf):
    if conf.table_type == "none":
        table = {}
    else:
        table = load_align_scores(conf.table_path)
    return table


def convert_doc_ids_integer(dl, inv_index):
    inv_index_i = {}
    for q_term, entries in inv_index.items():
        inv_index_i[q_term] = [(int(doc_id), cnt) for doc_id, cnt in entries]

    dl_i = {int(doc_id): n for doc_id, n in dl.items()}
    return dl_i, inv_index_i


def get_bm25t_retriever_in_memory(conf):
    table = load_table(conf)

    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_conf)
    dl, inv_index = convert_doc_ids_integer(dl, inv_index)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    tokenize_fn = get_tokenize_fn(bm25_conf)

    def get_posting(term):
        try:
            return inv_index[term]
        except KeyError:
            return []

    return BM25T_Retriever2(get_posting, df, dl, scoring_fn, tokenize_fn, table)


def get_bm25t_retriever_w_server(conf):
    client = InvIndexReaderClient()
    _ = client.get_postings("book")

    table = load_table(conf)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    tokenize_fn = get_tokenize_fn(bm25_conf)

    return BM25T_Retriever2(client.get_postings, df, dl, scoring_fn, tokenize_fn, table)


def main():
    c_log.setLevel(logging.INFO)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    retriever = get_bm25t_retriever_in_memory(conf)
    run_retrieval_eval_report_w_conf(conf, retriever)


if __name__ == "__main__":
    main()
