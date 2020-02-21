import os
import pickle
from collections import Counter

import nltk

from adhoc.bm25 import BM25_2
from data_generator.tokenizer_b import BasicTokenizer


class StemmerCache:
    def __init__(self, cache = None):
        self.stemmer = Stemmer()
        if cache is not None:
            self.cache = cache
        else:
            self.cache = dict()

    def stem(self, t):
        if t in self.cache:
            return self.cache[t]
        else:
            r = self.stemmer.stem(t)
            self.cache[t] = r
            if len(self.cache) % 1000 == 0:
                pickle.dump(self.cache, open("stemmer.pickle", "wb"))
            return r


def load_stemmer():
    if os.path.exists("stemmer.pickle"):
        cache = pickle.load(open("stemmer.pickle", "rb"))
    else:
        cache = None

    return StemmerCache(cache)


stemmer = load_stemmer()
tokenizer = BasicTokenizer(True)


def stem_tokenize(text):
    return list([stemmer.stem(t) for t in nltk.word_tokenize(text)])


mu = 1000


def get_bm25(query, doc, df, N, avdl):
    q_terms = stem_tokenize(query)
    d_terms = stem_tokenize(doc)
    q_tf = Counter(q_terms)
    d_tf = Counter(d_terms)
    score = 0
    dl = len(d_terms)
    for q_term in q_terms:
        #tf = (d_tf[q_term] *dl / (mu+dl) + ctf[q_term] * mu / (mu+dl))
        #score += score_BM25(n=df[q_term], f=tf, qf=q_tf[q_term], r=0, N=N,
        #                   dl=len(d_terms), avdl=avdl)
        score += BM25_2(d_tf[q_term], df[q_term], N, dl, avdl)
    return score