from omegaconf import OmegaConf

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.other.bm25_retriever_helper import get_tokenize_fn
from adhoc.other.bm25t_retriever import BM25T_Retriever2, IndexReaderPython
from dataset_specific.msmarco.passage.doc_indexing.retriever import load_bm25_resources
from models.classic.stopword import load_stopwords
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores
from typing import List, Iterable, Callable, Dict, Tuple, Set


def load_table(conf) -> Dict[str, Dict[str, float]]:
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

    index_reader = IndexReaderPython(get_posting, df, dl)
    stopwords = load_stopwords()
    return BM25T_Retriever2(index_reader, scoring_fn, tokenize_fn, table, stopwords)
