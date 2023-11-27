
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set


from adhoc.bm25 import BM25_verbose
from trainer_v2.chair_logging import c_log


class BM25T_3:
    def __init__(
            self,
            mapping: Dict[str, Dict[str, float]],
            bm25,
            tokenize_fn
    ):
        self.relevant_terms = mapping
        self.tokenize_fn = tokenize_fn

        self.df = bm25.df
        self.N = bm25.N
        self.avdl = bm25.avdl

        self.k1 = bm25.k1
        self.k2 = bm25.k2
        self.b = bm25.b
        self.n_mapping_used = 0
        self.mapping_used_flag = 0
        self.log_mapping = Counter()

    def __del__(self):
        matched_query = 0
        for k, v in self.log_mapping.items():
            if v:
                matched_query += 1

        c_log.info("%d queries used table", matched_query)
        c_log.info("%d query-document pairs used table", self.n_mapping_used)

    def score(self, query, text):
        q_terms = self.tokenize_fn(query)
        t_terms = self.tokenize_fn(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        self.mapping_used_flag = 0
        score = self.score_from_tfs(q_tf, t_tf)
        if self.mapping_used_flag:
            self.log_mapping[query] = 1

        return score

    def score_batch(self, qd_list) -> List[float]:
        output: List[float] = []
        for q, d in qd_list:
            s = self.score(q, d)
            output.append(s)
        return output

    def score_from_tfs(self, q_tf: Dict[str, int], t_tf: Dict[str, int]):
        dl = sum(t_tf.values())
        score_sum = 0
        for q_term, q_cnt in q_tf.items():
            translation_term_set: Dict[str, float] = self.relevant_terms[q_term]

            exact_cnt = t_tf[q_term]

            expansion_tf: List[float] = []
            for t, cnt in t_tf.items():
                if t in translation_term_set:
                    self.n_mapping_used += 1
                    expansion_tf.append(translation_term_set[t])

            if expansion_tf:
                self.mapping_used_flag = 1

            if exact_cnt:
                tf_sum = exact_cnt
            elif expansion_tf:
                tf_sum: float = max(expansion_tf)
            else:
                tf_sum = 0

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

            score_sum += t
        return score_sum


