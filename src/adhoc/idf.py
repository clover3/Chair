from collections import Counter
import math

class Idf_tokens:
    def __init__(self, docs):
        self.df = Counter()
        self.idf = dict()
        for doc in docs:
            term_count = Counter()
            for token in doc:
                term_count[token] = 1
            for elem, cnt in term_count.items():
                self.df[elem] += 1
        N = len(docs)

        for term, df in self.df.items():
            self.idf[term] = math.log(N/df)
        self.default_idf = math.log(N/1)

    def __getitem__(self, term):
        if term in self.idf:
            return self.idf[term]
        else:
            return self.default_idf
