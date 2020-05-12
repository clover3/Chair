import collections
from typing import Dict, Any

from models.classic.stopword import load_stopwords
from summarization.random_walk import run_random_walk, Edges


def count_co_ocurrence(window_size, raw_count, token_doc):
    for i in range(len(token_doc)):
        source = token_doc[i]
        st = max(i - int(window_size / 2), 0)
        ed = min(i + int(window_size / 2), len(token_doc))
        for j in range(st, ed):
            target = token_doc[j]
            raw_count[source][target] += 1


class TextRank:
    def __init__(self, data):
        self.p_reset = 0.1
        self.max_repeat = 500
        self.window_size = 10
        self.idf = collections.Counter()
        self.def_idf = 2
        for document in data:
            for word in set(document):
                self.idf[word] += 1
        self.stopword = load_stopwords()

    def get_edge(self, vertice, token_doc):
        raw_count = dict((vertex, collections.Counter()) for vertex in vertice)
        count_co_ocurrence(self.window_size, raw_count, token_doc)

        edges = dict()
        for vertex in vertice:
            out_sum = sum(raw_count[vertex].values())
            out_weights = dict()
            for target in raw_count.keys():
                out_weights[target] = raw_count[vertex][target] / out_sum
            edges[vertex] = out_weights
        return edges

    def get_reset(self, token_doc):
        tf_d = collections.Counter(token_doc)
        d = dict()
        for word, tf in tf_d.items():
            if word in self.idf:
                d[word] = tf / self.idf[word]
            else:
                d[word] = tf / self.def_idf
        total = sum(d.values())
        for word in tf_d.keys():
            d[word] = d[word] / total
        return d

    def run(self, raw_token_doc):
        def not_stopword(word):
            return not word in self.stopword
        token_doc = list(filter(not_stopword, raw_token_doc))
        vertice = set(token_doc)
        v_reset = self.get_reset(token_doc)

        edges: Dict[Any, Dict] = self.get_edge(vertice, token_doc)

        return run_random_walk(Edges(edges), vertice, self.max_repeat, self.p_reset)


