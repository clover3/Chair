import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from misc_lib import SuccessCounter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep


class PepUniRanker:
    def __init__(self,
                 table: Dict[str, Dict[str, float]],
                 bm25_scoring_fn: Callable[[int, int, int, int], float],
                 df: Dict[str, int],
                 ):
        self.table = table
        self.tokenizer = get_tokenizer()
        self.q_rep_cache = {}
        self.is_elite_cache = {}
        self.logged = set()

        self.missed_q_terms = set()
        self.matched_q_terms = set()
        self.per_q_max = {}
        self.per_q_min = {}
        for q_term, entries in table.items():
            self.per_q_max[q_term] = max([score for d_term, score in entries.items()])
            self.per_q_min[q_term] = min([score for d_term, score in entries.items()])

        self.qd_hit = SuccessCounter()

        self.bm25_scoring_fn: Callable[[int, int, int, int], float] = bm25_scoring_fn
        self.tokenizer = get_tokenizer()
        self.df = df

    def get_q_rep(self, query):
        if query in self.q_rep_cache:
            q_rep = self.q_rep_cache[query]
        else:
            q_rep = TextRep.from_text(self.tokenizer, query)
            self.q_rep_cache[query] = q_rep
        return q_rep

    def pair_score(self, q_term, d_term) -> float:
        if q_term in self.table:
            self.matched_q_terms.add(q_term)
            q_entry = self.table[q_term]
            if d_term in q_entry:
                s = q_entry[d_term]
                self.qd_hit.suc()
            else:
                s = self.per_q_min[q_term] - 0.01
                self.qd_hit.fail()
        else:
            self.missed_q_terms.add(q_term)
            s = 0
            self.qd_hit.fail()

        return s

    def score_from_text_rep(self, q_rep: TextRep, d_rep: TextRep) -> float:
        c_log.debug("Query: %s", " ".join(q_rep.tokenized_text.sp_tokens))
        c_log.debug("Doc: %s", " ".join(d_rep.tokenized_text.sp_tokens))

        score_sum = 0
        for q_term in q_rep.get_terms():
            if q_term in self.table:
                raw_tf = d_rep.counter[q_term]
                pair_scores = [self.pair_score(q_term, dt) for dt, dt_tf, _ in d_rep.get_bow()]
                s2 = max(pair_scores)
                if raw_tf:
                    if q_term not in self.per_q_max:
                        self.missed_q_terms.add(q_term)
                        s1 = 3
                    else:
                        s1 = max(self.per_q_max[q_term], 1)
                    score = max(s1, s2)
                else:
                    score = s2
            else:
                raw_tf = d_rep.counter[q_term]
                qf = q_rep.counter[q_term]
                qdf = self.df[q_term] if q_term in self.df else 0
                dl = d_rep.get_sp_size()
                score = self.bm25_scoring_fn(raw_tf, qf, dl, qdf)
            score_sum += score
        return score_sum

    def score(self, qd_iter: Iterable[Tuple[str, str]]) -> List[float]:
        def convert_to_reps(qd):
            q, d = qd
            q_rep = self.get_q_rep(q)
            d_rep = TextRep.from_text(self.tokenizer, d)
            return q_rep, d_rep

        qd_reps: List[Tuple[TextRep, TextRep]] = list(map(convert_to_reps, qd_iter))
        output: List[float] = [self.score_from_text_rep(q, d) for q, d in qd_reps]
        return output