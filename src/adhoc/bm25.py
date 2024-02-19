from math import log

k1 = 1.2
k2 = 100
k3 = 1
b = 0.75
R = 0.0


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

def BM25_3(f, qf, df, N, dl, avdl):
    K = compute_K(dl, avdl)
    first = log((N-df+0.5)/(df + 0.5))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2+1) * qf) / (k2 + qf)
    return first * second * third


def BM25_verbose(f, qf, df, N, dl, avdl, b, my_k1, my_k2):
    K: float = my_k1 * ((1-b) + b * (float(dl)/float(avdl)))
    first: float = log((N-df+0.5)/(df + 0.5) + 1)
    second: float = ((my_k1 + 1) * f) / (K + f)
    third: float = ((my_k2+1) * qf) / (my_k2 + qf)
    return first * second * third


def BM25_verbose_f(f, qf, df, N, dl, avdl, b, my_k1, my_k2):
    K = my_k1 * ((1. - b) + b * dl / avdl)
    first = log((N - df + 0.5) / (df + 0.5) + 1.)
    second = ((my_k1 + 1.) * f) / (K + f)
    third = ((my_k2 + 1.) * qf) / (my_k2 + qf)
    return first * second * third


def BM25_3_q_weight(qf, df, N):
    first = log((N-df+0.5)/(df + 0.5))
    third = ((k2+1) * qf) / (k2 + qf)
    return first * third


def BM25_reverse(score, df, N, dl, avdl):
    K = k1 * (1- b+b*dl / avdl)
    idf = log((N-df+0.5)/(df + 0.5))
    tf = (score - K) / ( (k1+1)* idf -score )
    return tf

def compute_K(dl, avdl):
    return k1 * ((1-b) + b * (float(dl)/float(avdl)) )


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