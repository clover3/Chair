import tensorflow as tf
import os

from adhoc.other.bm25_retriever_helper import get_stopwords_from_conf, get_tokenize_fn
from adhoc.other.index_reader_wrap import QLIndexReaderPython
from adhoc.other.ql.ql_formula import ql_scoring_inner
from adhoc.other.ql.ql_retriever import QLRetriever
from cache import load_pickle_from
from trainer_v2.chair_logging import c_log
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

def load_ql_resources(conf, avdl=None):
    if not os.path.exists(conf.inv_index_path):
        raise FileNotFoundError(conf.inv_index_path)
    if not os.path.exists(conf.bg_prob_path):
        raise FileNotFoundError(conf.bg_prob_path)
    if not os.path.exists(conf.dl_path):
        raise FileNotFoundError(conf.dl_path)
    c_log.info("Loading bg_prob from %s", conf.bg_prob_path)
    bg_prob = load_pickle_from(conf.bg_prob_path)
    c_log.info("Loading document length (dl) from %s", conf.dl_path)
    dl = load_pickle_from(conf.dl_path)
    c_log.info("Loading inv_index form %s", conf.inv_index_path)
    inv_index: Dict[str, List[Tuple[str, int]]] = load_pickle_from(conf.inv_index_path)
    c_log.info("Done")
    cdf = len(dl)
    if avdl is None:
        avdl = sum(dl.values()) / cdf
    return avdl, cdf, bg_prob, dl, inv_index


def load_ql_stats(conf, avdl=None):
    if not os.path.exists(conf.bg_prob_path):
        raise FileNotFoundError(conf.bg_prob_path)
    if not os.path.exists(conf.dl_path):
        raise FileNotFoundError(conf.dl_path)
    c_log.info("Loading bg_prob from %s", conf.bg_prob_path)
    bg_prob = load_pickle_from(conf.bg_prob_path)
    c_log.info("Loading document length (dl) from %s", conf.dl_path)
    dl = load_pickle_from(conf.dl_path)
    c_log.info("Done")
    cdf = len(dl)
    if avdl is None:
        avdl = sum(dl.values()) / cdf
    return avdl, cdf, bg_prob, dl


def build_ql_scoring_fn_from_conf(conf):
    mu = conf.mu

    def ql_scoring(tf, qf, dl, bg_prob_val):
        return ql_scoring_inner(tf, qf, dl, bg_prob_val, mu)
    return ql_scoring


def get_ql_retriever_from_conf(conf) -> QLRetriever:
    stopwords = get_stopwords_from_conf(conf)
    _avdl, _cdf, bg_prob, dl, inv_index = load_ql_resources(conf)
    tokenize_fn = get_tokenize_fn(conf)

    def get_posting(term):
        try:
            return inv_index[term]
        except KeyError:
            return []

    index_reader = QLIndexReaderPython(get_posting, bg_prob, dl)
    scoring_fn = build_ql_scoring_fn_from_conf(conf)
    return QLRetriever(
        index_reader,
        scoring_fn,
        tokenize_fn,
        stopwords)
