
# BM25 with table-based expansion
#
#
#
#

from collections import Counter
from dataclasses import dataclass
from typing import Dict

from adhoc.bm25 import BM25_verbose
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer


class BM25T:
    def __init__(self, mapping: Dict[str, Dict[str, float]],
                 bm25):
        self.mapping = mapping
        self.tokenizer = KrovetzNLTKTokenizer()

        self.df = bm25.df
        self.N = bm25.N
        self.avdl = bm25.avdl

        self.k1 = bm25.k1
        self.k2 = bm25.k2
        self.b = bm25.b
        self.n_mapping_used = 0

    def score(self, query, text):
        q_terms = self.tokenizer.tokenize_stem(query)
        t_terms = self.tokenizer.tokenize_stem(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        dl = sum(t_tf.values())
        score_sum = 0
        for q_term, q_cnt in q_tf.items():
            translation_term_set: Dict[str, float] = self.mapping[q_term]
            expansion_tf = 0
            for t, cnt in t_tf.items():
                if t in translation_term_set:
                    self.n_mapping_used += 1
                    expansion_tf += cnt * translation_term_set[t]
                    # c_log.debug(f"matched {t} has {translation_term_set[t]}")


            raw_cnt = t_tf[q_term]
            tf_sum = expansion_tf + raw_cnt

            t = BM25_verbose(f=tf_sum,
                         qf=q_cnt,
                         df=self.df[q_term],
                         N=self.N,
                         dl=dl,
                         avdl=self.avdl,
                         b=self.b,
                         my_k1=self.k1,
                         my_k2=self.k2
                         )
            # if expansion_tf:
            #     c_log.debug(f"tf_sum={expansion_tf}+{raw_cnt}, adding {t} to total")

            score_sum += t
        return score_sum


@dataclass
class GlobalAlign:
    token_id: int
    word: str
    score: float
    n_appear: int
    n_pos_appear: int


