from typing import List, Dict, Tuple, Iterable

from adhoc.build_index import build_inverted_index
from cache import save_to_pickle
from dataset_specific.msmarco.passage.doc_indexing.resource_loader import enum_msmarco_passage_tokenized
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from models.classic.stopword import load_stopwords

InvIndex = Dict[str, List[Tuple[str, int]]]
IntInvIndex = Dict[str, List[Tuple[int, int]]]


def mmp_inv_index_ignore_voca():
    cdf, df = load_msmarco_passage_term_stat()
    voca = []
    for term, cnt in df.items():
        portion = cnt / cdf
        if portion > 0.1:
            voca.append(term)

    voca.extend(load_stopwords())
    return voca


def msmarco_build_inverted_index() -> InvIndex:
    term_df_cut_to_discard = 1000000
    term_df_cut_to_warn = 10000
    num_docs = 8841823
    tokenized_itr: Iterable[Tuple[str, List[str]]] = enum_msmarco_passage_tokenized()

    ignore_voca = set(mmp_inv_index_ignore_voca())
    return build_inverted_index(tokenized_itr, ignore_voca, num_docs, term_df_cut_to_discard, term_df_cut_to_warn)


def main():
    inv_index = msmarco_build_inverted_index()
    save_to_pickle(inv_index, "mmp_inv_index_int_krovetz")


if __name__ == "__main__":
    main()