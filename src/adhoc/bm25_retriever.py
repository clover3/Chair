from abc import ABC, abstractmethod
from collections import Counter

from adhoc.bm25 import BM25_verbose
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from list_lib import left
from misc_lib import get_second
from typing import List, Iterable, Tuple


class RetrieverIF(ABC):
    @abstractmethod
    def retrieve(self, query) -> List[Tuple[str, float]]:
        pass


class BM25Retriever(RetrieverIF):
    def __init__(self, tokenize_fn, inv_index, df, dl_d, scoring_fn, stopwords=None):
        self.inv_index = inv_index
        self.scoring_fn = scoring_fn
        self.df = df
        self.tokenize_fn = tokenize_fn
        self.dl_d = dl_d
        if stopwords is not None:
            self.stopwords = stopwords
        else:
            self.stopwords = set()

    def get_low_df_terms(self, q_terms: Iterable[str], n_limit=100) -> List[str]:
        candidates = []
        for t in q_terms:
            df = self.df[t]
            candidates.append((t, df))

        candidates.sort(key=get_second)
        return left(candidates)[:n_limit]

    def get_posting(self, term):
        if term in self.inv_index:
            return self.inv_index[term]
        else:
            return []

    def retrieve(self, query) -> List[Tuple[str, float]]:
        q_tokens = self.tokenize_fn(query)
        q_tokens = [t for t in q_tokens if not t in self.stopwords]
        q_tf = Counter(q_tokens)
        doc_score = Counter()
        indexing_q_terms: List[str] = self.get_low_df_terms(q_tf.keys())
        non_indexing_q_terms = [term for term in q_tf.keys() if term not in indexing_q_terms]
        for term in indexing_q_terms:
            qf = q_tf[term]
            postings = self.get_posting(term)
            qdf = len(postings)
            for doc_id, cnt in postings:
                tf = cnt
                dl = self.dl_d[doc_id]
                per_q_term_score = self.scoring_fn(tf, qf, dl, qdf)
                doc_score[doc_id] += per_q_term_score

        for term in non_indexing_q_terms:
            qf = q_tf[term]
            postings = self.get_posting(term)
            qdf = len(postings)
            for doc_id, cnt in postings:
                if doc_id in doc_score:
                    tf = cnt
                    dl = self.dl_d[doc_id]
                    per_q_term_score = self.scoring_fn(tf, qf, dl, qdf)
                    doc_score[doc_id] += per_q_term_score

        return list(doc_score.items())


class BM25RetrieverKNTokenize(BM25Retriever):
    def __init__(self, inv_index, df, dl_d, scoring_fn, stopwords=None):
        self.tokenizer = KrovetzNLTKTokenizer(False)
        self.tokenize_fn = self.tokenizer.tokenize_stem
        super(BM25RetrieverKNTokenize, self).__init__(
            self.tokenizer.tokenize_stem, inv_index, df, dl_d, scoring_fn, stopwords)


def build_bm25_scoring_fn(cdf, avdl):
    b = 1.2
    k1 = 0.1
    k2 = 100

    def scoring_fn(tf, qf, dl, qdf):
        return BM25_verbose(tf, qf, qdf, cdf, dl, avdl, b, k1, k2)

    return scoring_fn
