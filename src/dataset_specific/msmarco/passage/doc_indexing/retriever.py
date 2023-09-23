from adhoc.bm25_retriever import BM25Retriever, build_bm25_scoring_fn
from cache import load_from_pickle
from dataset_specific.msmarco.passage.doc_indexing.build_inverted_index_msmarco import InvIndex
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat


def get_retriever():
    inv_index: InvIndex = load_from_pickle("mmp_inv_index_int_krovetz")
    cdf, df = load_msmarco_passage_term_stat()
    scoring_fn = build_bm25_scoring_fn(cdf, 25)
    dl_d = load_from_pickle("msmarco_passage_dl_d")
    return BM25Retriever(inv_index, df, dl_d, scoring_fn)


if __name__ == "__main__":
    get_retriever()
