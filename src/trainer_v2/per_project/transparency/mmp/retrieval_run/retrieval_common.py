from omegaconf import OmegaConf

from adhoc.other.bm25_retriever_helper import get_tokenize_fn, \
    get_stopwords_from_conf, build_bm25_scoring_fn_from_conf, load_bm25_resources
from adhoc.other.bm25t_retriever import BM25T_Retriever2, BM25T_Retriever3
from adhoc.other.index_reader_wrap import IndexReaderPython
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores, load_mapping_from_align_scores
from typing import List, Iterable, Callable, Dict, Tuple, Set


def load_table(conf) -> Dict[str, Dict[str, float]]:
    if conf.table_type == "none" or conf.table_path == "none":
        table = {}
    elif conf.table_type == "mapping":
        cut = 0
        mapping_val = 0.1
        table = load_mapping_from_align_scores(conf.table_path, cut, mapping_val)
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
    return get_bm25t_in_memory_inner(bm25_conf, table)


def get_bm25t_in_memory_inner(bm25_conf, table):
    stopwords = get_stopwords_from_conf(bm25_conf)
    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_conf, None)
    tokenize_fn = get_tokenize_fn(bm25_conf)
    scoring_fn = build_bm25_scoring_fn_from_conf(bm25_conf, avdl, cdf)

    def get_posting(term):
        try:
            return inv_index[term]
        except KeyError:
            return []

    index_reader = IndexReaderPython(get_posting, df, dl)
    return BM25T_Retriever2(index_reader, scoring_fn, tokenize_fn, table, stopwords)


def get_bm25t_retriever3(conf):
    table = load_table(conf)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    stopwords = get_stopwords_from_conf(bm25_conf)
    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_conf, None)
    tokenize_fn = get_tokenize_fn(bm25_conf)
    scoring_fn = build_bm25_scoring_fn_from_conf(bm25_conf, avdl, cdf)

    def get_posting(term):
        try:
            return inv_index[term]
        except KeyError:
            return []

    index_reader = IndexReaderPython(get_posting, df, dl)
    return BM25T_Retriever3(index_reader, scoring_fn, tokenize_fn, table, stopwords)
