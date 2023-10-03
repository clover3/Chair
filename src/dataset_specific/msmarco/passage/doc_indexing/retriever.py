import os.path
from typing import List, Iterable, Callable, Dict, Tuple, Set

from adhoc.bm25_retriever import BM25Retriever, build_bm25_scoring_fn
from cache import load_from_pickle, load_pickle_from
from dataset_specific.msmarco.passage.doc_indexing.build_inverted_index_msmarco import InvIndex
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_bm25_no_stem_resource_path_helper, \
    get_bm25_stem_resource_path_helper
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from models.classic.stopword import get_all_punct
from trainer_v2.chair_logging import c_log


def get_mmp_bm25_retriever():
    inv_index: InvIndex = load_from_pickle("mmp_inv_index_krovetz2")
    cdf, df = load_msmarco_passage_term_stat()
    scoring_fn = build_bm25_scoring_fn(cdf, 52)
    dl_d = load_from_pickle("msmarco_passage_dl_d")
    return BM25Retriever(inv_index, df, dl_d, scoring_fn)


def get_mmp_bm25_retriever_stemmed():
    conf = get_bm25_stem_resource_path_helper()
    avdl = 52
    return get_bm25_retriever_from_conf(conf, avdl)


def get_mmp_bm25_retriever_stemmed_stop_puntc():
    conf = get_bm25_stem_resource_path_helper()
    avdl = 52
    stopwords = set(get_all_punct())
    return get_bm25_retriever_from_conf(conf, avdl, stopwords)


def get_mmp_bm25_retriever_no_stem():
    conf = get_bm25_no_stem_resource_path_helper()
    avdl = 52
    return get_bm25_retriever_from_conf(conf, avdl)


def get_bm25_retriever_from_conf(conf, avdl=None, stopwords=None) -> BM25Retriever:
    avdl, cdf, df, dl, inv_index = load_bm25_resources(conf, avdl)

    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    return BM25Retriever(inv_index, df, dl, scoring_fn, stopwords)


def load_bm25_resources(conf, avdl=None):
    if not os.path.exists(conf.inv_index_path):
        raise FileNotFoundError(conf.inv_index_path)
    if not os.path.exists(conf.df_path):
        raise FileNotFoundError(conf.df_path)
    if not os.path.exists(conf.dl_path):
        raise FileNotFoundError(conf.dl_path)
    c_log.info("Loading document frequency (df)")
    df = load_pickle_from(conf.df_path)
    c_log.info("Loading document length (dl)")
    dl = load_pickle_from(conf.dl_path)
    c_log.info("Loading inv_index")
    inv_index: InvIndex = load_pickle_from(conf.inv_index_path)
    c_log.info("Done")
    cdf = len(dl)
    if avdl is None:
        avdl = sum(dl.values()) / cdf
    return avdl, cdf, df, dl, inv_index


def get_bm25_stats_from_conf(conf, avdl=None) -> Tuple:
    if not os.path.exists(conf.df_path):
        raise FileNotFoundError(conf.df_path)
    if not os.path.exists(conf.dl_path):
        raise FileNotFoundError(conf.dl_path)

    c_log.info("Loading document frequency (df)")
    df = load_pickle_from(conf.df_path)
    c_log.info("Loading document length (dl)")
    dl = load_pickle_from(conf.dl_path)
    c_log.info("Done")
    cdf = len(dl)
    if avdl is None:
        avdl = sum(dl.values()) / cdf
    return avdl, cdf, df, dl


if __name__ == "__main__":
    get_mmp_bm25_retriever()
