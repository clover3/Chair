from collections import Counter
from typing import List, Tuple

from arg.bm25 import BM25
from list_lib import flatten


class BM25Scorer:
    def __init__(self, bm25: BM25):
        self.bm25 = bm25

    def predict(self, query_passage_list: List[Tuple[str, List[List[str]]]]) -> List[float]:
        score_list = []
        for query, passage in query_passage_list:
            score = self.score(query, passage)
            score_list.append(score)
        return score_list

    def score(self, query, token_list_list):
        q_terms = self.bm25.tokenizer.tokenize_stem(query)
        tokens = flatten(token_list_list)
        t_terms = list(map(self.bm25.tokenizer.stemmer.stem, tokens))
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        score = self.bm25.score_inner(q_tf, t_tf)
        return score

