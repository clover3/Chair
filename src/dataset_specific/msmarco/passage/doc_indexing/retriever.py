from adhoc.bm25_retriever import BM25RetrieverKNTokenize, build_bm25_scoring_fn
from adhoc.other.bm25_retriever_helper import load_bm25_resources
from cache import load_from_pickle
from dataset_specific.msmarco.passage.doc_indexing.build_inverted_index_msmarco import InvIndex
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_bm25_no_stem_resource_path_helper, \
    get_bm25_stem_resource_path_helper
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from models.classic.stopword import get_all_punct


def get_mmp_bm25_retriever():
    inv_index: InvIndex = load_from_pickle("mmp_inv_index_krovetz2")
    cdf, df = load_msmarco_passage_term_stat()
    scoring_fn = build_bm25_scoring_fn(cdf, 52)
    dl_d = load_from_pickle("msmarco_passage_dl_d")
    return BM25RetrieverKNTokenize(inv_index, df, dl_d, scoring_fn)


def get_mmp_bm25_retriever_stemmed():
    conf = get_bm25_stem_resource_path_helper()
    avdl = 52
    return get_kn_bm25_retriever_from_conf(conf, avdl)


def get_mmp_bt1_bm25_retriever():
    conf = get_bm25_stem_resource_path_helper()
    avdl = 52
    return get_kn_bm25_retriever_from_conf(conf, avdl)


def get_mmp_bm25_retriever_stemmed_stop_puntc():
    conf = get_bm25_stem_resource_path_helper()
    avdl = 52
    stopwords = set(get_all_punct())
    return get_kn_bm25_retriever_from_conf(conf, avdl, stopwords)


def get_mmp_bm25_retriever_no_stem():
    conf = get_bm25_no_stem_resource_path_helper()
    avdl = 52
    return get_kn_bm25_retriever_from_conf(conf, avdl)


def get_kn_bm25_retriever_from_conf(conf, avdl=None, stopwords=None) -> BM25RetrieverKNTokenize:
    avdl, cdf, df, dl, inv_index = load_bm25_resources(conf, avdl)

    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    return BM25RetrieverKNTokenize(inv_index, df, dl, scoring_fn, stopwords)


if __name__ == "__main__":
    get_mmp_bm25_retriever()
