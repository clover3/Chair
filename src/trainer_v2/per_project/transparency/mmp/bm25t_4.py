
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set


from adhoc.bm25 import BM25_verbose
from trainer_v2.chair_logging import c_log


class BM25T_4:
    def __init__(
            self,
            mapping: Dict[str, Dict[str, float]],
            scoring_fn,
            tokenize_fn,
            df
    ):
        self.relevant_terms = mapping
        self.tokenize_fn = tokenize_fn

        self.df = df
        self.scoring_fn = scoring_fn
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
            exact_cnt = t_tf[q_term]

            if exact_cnt:
                tf_sum = exact_cnt
            else:
                translation_term_set: Dict[str, float] = self.relevant_terms[q_term]
                expansion_tf: List[float] = []
                for t, cnt in t_tf.items():
                    if t in translation_term_set:
                        self.n_mapping_used += 1
                        expansion_tf.append(translation_term_set[t])

                if expansion_tf:
                    self.mapping_used_flag = 1

                if expansion_tf:
                    tf_sum: float = max(expansion_tf)
                else:
                    tf_sum = 0

            score_sum += self.scoring_fn(tf_sum, q_cnt, dl, self.df[q_term])
        return score_sum


