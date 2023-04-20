from collections import Counter
from typing import Dict, List

from adhoc.bm25 import BM25_verbose
from adhoc.bm25_class import BM25
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer


class BM25TFeatureEncoder:
    def __init__(self, bm25: BM25, candidate_voca):
        self.bm25_bare = bm25.core
        self.tokenizer = KrovetzNLTKTokenizer()
        self.target_q_term = "when"
        self.candidate_voca: Dict = candidate_voca

    def get_term_translation_weight_feature(self, query: str, text: str):
        q_terms = self.tokenizer.tokenize_stem(query)
        t_terms = self.tokenizer.tokenize_stem(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        dl = sum(t_tf.values())

        score = 0
        d = {}
        for q_term, q_cnt in q_tf.items():
            if q_term == self.target_q_term:
                for t, t_cnt in t_tf.items():
                    if t in self.candidate_voca:
                        token_id = self.candidate_voca[t]
                        d[token_id] = t_cnt
            else:
                tf_sum = t_tf[q_term]
                t = BM25_verbose(f=tf_sum,
                                 qf=q_cnt,
                                 df=self.bm25_bare.df[q_term],
                                 N=self.bm25_bare.N,
                                 dl=dl,
                                 avdl=self.bm25_bare.avdl,
                                 b=self.bm25_bare.b,
                                 my_k1=self.bm25_bare.k1,
                                 my_k2=self.bm25_bare.k2
                                 )
                score += t

        feature_ids: List[str] = list(d.keys())
        feature_values: List[float] = list(d.values())
        return score, feature_ids, feature_values