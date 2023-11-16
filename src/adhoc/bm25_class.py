from collections import Counter

from math import log

from adhoc.bm25 import BM25_verbose
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer


class BM25Bare:
    def __init__(self, df, num_doc, avdl, k1=0.01, k2=100, b=0.6):
        self.N = num_doc
        self.avdl = avdl
        self.k1 = k1
        self.k2 = k2
        self.df = df
        self.b = b

    def term_idf_factor(self, term):
        N = self.N
        df = self.df[term]
        return log((N - df + 0.5) / (df + 0.5))

    def score_inner(self, q_tf, t_tf) -> float:
        dl = sum(t_tf.values())
        score_sum = 0
        info = []
        for q_term, qtf in q_tf.items():
            t = BM25_verbose(f=t_tf[q_term],
                         qf=qtf,
                         df=self.df[q_term],
                         N=self.N,
                         dl=dl,
                         avdl=self.avdl,
                         b=self.b,
                         my_k1=self.k1,
                         my_k2=self.k2
                         )
            score_sum += t
            info.append((q_term, t))

        return score_sum


class BM25:
    def __init__(self, df, num_doc, avdl, k1=0.01, k2=100, b=0.6,
                 drop_stopwords=False):
        self.core = BM25Bare(df, num_doc, avdl, k1, k2, b)
        self.tokenizer = KrovetzNLTKTokenizer(drop_stopwords)

    def score(self, query, text) -> float:
        q_terms = self.tokenizer.tokenize_stem(query)
        t_terms = self.tokenizer.tokenize_stem(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        return self.core.score_inner(q_tf, t_tf)

    def term_idf_factor(self, term):
        return self.core.term_idf_factor(term)

    def score_inner(self, q_tf, t_tf) -> float:
        return self.core.score_inner(q_tf, t_tf)


class BM25FromTokenizeFn:
    def __init__(self, tokenize_fn, df, num_doc, avdl, k1=0.01, k2=100, b=0.6,
                 ):
        self.core = BM25Bare(df, num_doc, avdl, k1, k2, b)
        self.tokenize_fn = tokenize_fn

    def score(self, query, text) -> float:
        q_terms = self.tokenize_fn(query)
        t_terms = self.tokenize_fn(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        return self.core.score_inner(q_tf, t_tf)

    def term_idf_factor(self, term):
        return self.core.term_idf_factor(term)

    def score_inner(self, q_tf, t_tf) -> float:
        return self.core.score_inner(q_tf, t_tf)