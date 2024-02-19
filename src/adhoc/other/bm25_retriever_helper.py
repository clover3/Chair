import os

import omegaconf
from krovetzstemmer import Stemmer
from adhoc.bm25_class import BM25FromTokenizeFn
from adhoc.bm25_retriever import BM25Retriever, build_bm25_scoring_fn
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from cache import load_pickle_from
from data_generator.tokenizer_wo_tf import get_tokenizer
from typing import List, Callable, Tuple, Dict
from trainer_v2.chair_logging import c_log


def get_bm25_retriever_from_conf(conf, avdl=None) -> BM25Retriever:
    stopwords = get_stopwords_from_conf(conf)
    avdl, cdf, df, dl, inv_index = load_bm25_resources(conf, avdl)
    tokenize_fn = get_tokenize_fn(conf)
    scoring_fn = build_bm25_scoring_fn_from_conf(conf, avdl, cdf)
    return BM25Retriever(tokenize_fn, inv_index, df, dl, scoring_fn, stopwords)


def get_stopwords_from_conf(conf):
    try:
        f = open(conf.stopword_path, "r")
        stopwords = {line.strip() for line in f}
    except omegaconf.errors.ConfigAttributeError:
        stopwords = set()
    return stopwords


def build_bm25_scoring_fn_from_conf(conf, avdl, cdf):
    try:
        scoring_fn = build_bm25_scoring_fn(cdf, avdl, conf.b, conf.k1, conf.k2)
    except KeyError:
        scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    return scoring_fn


def get_tokenize_fn(conf) -> Callable[[str], List[str]]:
    tokenizer_name = conf.tokenizer
    return get_tokenize_fn2(tokenizer_name)


def get_tokenize_fn2(tokenizer_name):
    if tokenizer_name == "KrovetzNLTK":
        tokenizer = KrovetzNLTKTokenizer()
        return tokenizer.tokenize_stem
    elif tokenizer_name == "BertTokenize1":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer.tokenize
    elif tokenizer_name == "BertTokenize2":
        tokenizer = get_tokenizer()
        return tokenizer.basic_tokenizer.tokenize
    elif tokenizer_name == "BertTokenize2+Stem":
        tokenizer = get_tokenizer()
        stemmer = Stemmer()
        def tokenize(text):
            tokens = tokenizer.basic_tokenizer.tokenize(text)
            return [stemmer.stem(t) for t in tokens]
        return tokenize
    elif tokenizer_name == "space":
        def tokenize(text):
            return text.split()
        return tokenize
    elif tokenizer_name == "lucene":
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        analyzer = Analyzer(get_lucene_analyzer())
        return analyzer.analyze
    elif tokenizer_name == "lucene_krovetz":
        from pyserini.analysis import Analyzer, get_lucene_analyzer
        analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))
        return analyzer.analyze
    else:
        raise ValueError(f"{tokenizer_name} is not expected")



def get_bm25_scorer_from_conf(conf, avdl=None) -> BM25FromTokenizeFn:
    avdl, cdf, df, dl = get_bm25_stats_from_conf(conf, avdl)
    tokenize_fn = get_tokenize_fn(conf)
    return BM25FromTokenizeFn(
        tokenize_fn, df, len(dl), avdl)

##
def load_bm25_resources(conf, avdl=None):
    if not os.path.exists(conf.inv_index_path):
        raise FileNotFoundError(conf.inv_index_path)
    if not os.path.exists(conf.df_path):
        raise FileNotFoundError(conf.df_path)
    if not os.path.exists(conf.dl_path):
        raise FileNotFoundError(conf.dl_path)
    c_log.info("Loading document frequency (df) from %s", conf.df_path)
    df = load_pickle_from(conf.df_path)
    c_log.info("Loading document length (dl) from %s", conf.dl_path)
    dl = load_pickle_from(conf.dl_path)
    c_log.info("Loading inv_index form %s", conf.inv_index_path)
    inv_index: Dict[str, List[Tuple[str, int]]] = load_pickle_from(conf.inv_index_path)
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
