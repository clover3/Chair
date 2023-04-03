from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from adhoc.bm25 import BM25_3
from adhoc.bm25_class import BM25
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from cache import load_from_pickle
from dataset_specific.msmarco.passage.doc_indexing.build_inverted_index_msmarco import InvIndex
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from list_lib import left
from misc_lib import get_second


class MMPRetrieval:
    def __init__(self, inv_index, df, cdf, avdl):
        self.inv_index = inv_index
        self.cdf = cdf
        self.avdl = avdl
        self.df = df
        self.tokenizer = KrovetzNLTKTokenizer(True)

    def get_low_df_terms(self, q_terms: Iterable[str], n_limit=10) -> Iterable[str]:
        candidates = []
        for t in q_terms:
            df = self.df[q_terms]
            candidates.append((t, df))

        candidates.sort(key=get_second)
        return left(candidates)[:n_limit]

    def get_posting(self, term):
        if term in self.inv_index:
            return self.inv_index[term]
        else:
            return []


    def retrieve(self, query):
        q_tokens = self.tokenizer.tokenize_stem(query)
        q_tf = Counter(q_tokens)
        doc_score = Counter()
        q_terms = self.get_low_df_terms(q_tf)
        for term in q_terms:
            qf = q_tf[term]
            postings = self.get_posting(term)
            qdf = len(postings)
            for doc_id, loc_list in postings:
                tf = len(loc_list)
                dl = self.avdl
                total_doc = self.cdf
                doc_score[doc_id] += BM25_3(tf, qf, qdf, total_doc, dl, self.avdl)




def get_retriever():
    inv_index: InvIndex = load_from_pickle("mmp_inv_index_int_krovetz")
    cdf, df = load_msmarco_passage_term_stat()
    return MMPRetrieval(inv_index, df, cdf, 25)


if __name__ == "__main__":
    get_retriever()
