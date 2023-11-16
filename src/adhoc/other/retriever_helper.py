from transformers import AutoTokenizer

from adhoc.bm25_class import BM25FromTokenizeFn
from adhoc.bm25_retriever import BM25Retriever, build_bm25_scoring_fn
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from dataset_specific.msmarco.passage.doc_indexing.retriever import load_bm25_resources, get_bm25_stats_from_conf
from typing import List, Callable


def get_bm25_retriever_from_conf(conf, avdl=None, stopwords=None) -> BM25Retriever:
    avdl, cdf, df, dl, inv_index = load_bm25_resources(conf, avdl)
    tokenize_fn = get_tokenize_fn(conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    return BM25Retriever(tokenize_fn, inv_index, df, dl, scoring_fn, stopwords)


def get_tokenize_fn(conf) -> Callable[[str], List[str]]:
    if conf.tokenizer == "KrovetzNLTK":
        tokenizer = KrovetzNLTKTokenizer()
        return tokenizer.tokenize_stem
    elif conf.tokenizer == "BertTokenize1":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer.tokenize
    else:
        pass


def get_bm25_scorer_from_conf(conf, avdl=None) -> BM25FromTokenizeFn:
    avdl, cdf, df, dl = get_bm25_stats_from_conf(conf, avdl)
    tokenize_fn = get_tokenize_fn(conf)
    return BM25FromTokenizeFn(
        tokenize_fn, df, len(dl), avdl)

