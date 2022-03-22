from typing import List, Callable, Tuple

import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from datastore.cached_client import MemoryCachedClient


class SentenceBertSolver(TokenScoringSolverIF):
    def __init__(self, encode_fn: Callable[[List[str]], List[np.array]]):
        self.encode_fn = encode_fn

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.get_max_similarity_rev(text1_tokens, text2_tokens)
        scores2 = self.get_max_similarity_rev(text2_tokens, text1_tokens)
        return scores1, scores2

    def get_max_similarity(self, text1_tokens, text2_tokens) -> List[float]:
        l1 = len(text1_tokens)
        l2 = len(text2_tokens)
        emb_list1 = self.encode_fn(text1_tokens)
        emb_list2 = self.encode_fn(text2_tokens)

        scores = []
        for i1 in range(l1):
            similarity_list: List[float] = [self.similar(emb_list1[i1], emb_list2[i2]) for i2 in range(l2)]
            scores.append(max(similarity_list))
        return scores

    def similar(self, emb1, emb2) -> float:
        try:
            return 1 - spatial.distance.cosine(emb1, emb2)
        except KeyError:
            return 0

    def get_max_similarity_rev(self, text1_tokens, text2_tokens) -> List[float]:
        scores = self.get_max_similarity(text1_tokens, text2_tokens)
        return [1-s for s in scores]


def get_sentence_bert_solver():
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    cache_client = MemoryCachedClient(model.encode, str, {})
    def predict_one(item):
        result_list = cache_client.predict([item])
        return result_list[0]

    return SentenceBertSolver(predict_one)
