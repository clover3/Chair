from collections import Counter
from math import log
from data_generator.tokenizer_b import BasicTokenizer
from krovetzstemmer import Stemmer
k1 = 1.2
k2 = 100
k3 = 1
b = 0.75
R = 0.0

stemmer = Stemmer()
def score_BM25(n, f, qf, r, N, dl, avdl):
    K = compute_K(dl, avdl)
    first = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2+1) * qf) / (k2 + qf)
    return first * second * third

def BM25_2(f, df, N, dl, avdl):
    first = (k1 + 1) * f / ( k1 * (1-b+b* dl / avdl ) + f)
    second = log((N-df+0.5)/(df + 0.5))
    return first * second

def compute_K(dl, avdl):
    return k1 * ((1-b) + b * (float(dl)/float(avdl)) )

tokenizer = BasicTokenizer(True)
def stem_tokenize(text):
    return list([stemmer.stem(t) for t in tokenizer.tokenize(text)])

mu = 1000
def get_bm25(query, doc, ctf, df, N, avdl):
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

import operator



class QueryProcessor:
    def __init__(self, queries, corpus):
        self.queries = queries
        self.index, self.dlt = NotImplemented #build_data_structures(corpus)

    def run(self):
        results = []
        for query in self.queries:
            results.append(self.run_query(query))
        return results

    def run_query(self, query):
        query_result = dict()
        for term in query:
            if term in self.index:
                doc_dict = self.index[term] # retrieve index entry
                for docid, freq in doc_dict.iteritems(): #for each document and its word frequency
                    score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
                                       dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
                    if docid in query_result: #this document has already been scored once
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        return query_result