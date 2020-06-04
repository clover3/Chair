from collections import Counter
from typing import Dict

import spacy

from arg.query_weight.query_weight_gen import QueryWeightGenInterface
from cache import save_to_pickle, load_from_pickle


class QueryWeightGenBySyntactic(QueryWeightGenInterface):
    def __init__(self, k=0.4, stemmer=None):
        super(QueryWeightGenBySyntactic, self).__init__()
        self.nlp_module = spacy.load("en_core_web_sm")
        self.stemmer = stemmer
        self.k = k
        self.cache_nlp_parse = {}

    def nlp(self, text):
        if text in self.cache_nlp_parse:
            return self.cache_nlp_parse[text]
        r = self.nlp_module(text)
        self.cache_nlp_parse[text] = r
        return r

    def save_nlp_cache(self, name=None):
        if name is None :
            name = "nlp_cache2"
        save_to_pickle(self.cache_nlp_parse, name)

    def load_nlp_cache_if_exist(self, name=None):
        if name is None:
            name = "nlp_cache2"
        try:
            self.cache_nlp_parse = load_from_pickle(name)
        except FileNotFoundError:
            pass

    def __del__(self):
        self.save_nlp_cache()

    def tokenize_stem(self, text):
        l = []
        for token in self.nlp(text):
            try:
                l.append(self.get_term_rep(token.text))
            except UnicodeDecodeError:
                pass
        return l

    def get_term_rep(self, raw_token):
        term_rep = raw_token
        if self.stemmer is not None:
            term_rep = self.stemmer.stem(raw_token)
        return term_rep

    # Terms in noun chunk get additional weight
    def gen_inner(self, text) -> Dict[str, float]:
        doc = self.nlp(text)

        tf = Counter()

        for token in doc:

            try:
                term_rep = self.get_term_rep(token.text)
                tf[term_rep] = 1
            except UnicodeDecodeError:
                pass

        k1_set = set()
        for chunk in doc.noun_chunks:
            for i in range(chunk.start, chunk.end):
                try:
                    term_rep = self.get_term_rep(doc[i].text)
                    k1_set.add(term_rep)
                except UnicodeDecodeError:
                    pass

        for term in k1_set:
            tf[term] += self.k
        return tf
