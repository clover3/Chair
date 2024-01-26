import collections
import dataclasses
import math
from abc import ABC, abstractmethod

from misc_lib import MovingWindow
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import get_pep_predictor


@dataclasses.dataclass
class TermAlignInfo:
    q_term: str
    d_term: str
    multiplier: float


class BM25_MatchAnalyzerIF(ABC):
    @abstractmethod
    def apply(self, q, d) -> tuple[float, list[TermAlignInfo], float]:
        pass


class BM25_MatchAnalyzer(BM25_MatchAnalyzerIF):
    def __init__(self, bm25, get_pep_top_k, tokenize_fn):
        self.get_pep_top_k = get_pep_top_k
        self.bm25 = bm25
        self.tokenize_fn = tokenize_fn
        self.avg_match_count = MovingWindow(200)

    def apply(self, q: str, d: str) -> tuple[float, list[TermAlignInfo], float]:
        bm25 = self.bm25
        q_tokens: list[str] = self.tokenize_fn(q)
        d_tokens: list[str] = self.tokenize_fn(d)

        q_counter = collections.Counter(q_tokens)
        d_counter = collections.Counter(d_tokens)

        def query_factor(q_term, qf) -> float:
            N = bm25.N
            df = bm25.df[q_term]
            idf_like = math.log((N - df + 0.5) / (df + 0.5) + 1)
            qft_based = ((bm25.k2 + 1) * qf) / (bm25.k2 + qf)
            return idf_like * qft_based

        dl = len(d_tokens)
        denom_factor = (1 + bm25.k1)
        K = bm25.k1 * ((1 - bm25.b) + bm25.b * (float(dl) / float(bm25.avdl)))
        per_unknown_tf: list[TermAlignInfo] = []
        n_exact_match = 0
        value_score = 0.0
        for q_term, qtf in q_counter.items():
            exact_match_cnt: int = d_counter[q_term]
            top_k_terms: list[str] = self.get_pep_top_k(q_term, d_counter.keys())

            if exact_match_cnt:
                score_per_q_term: float = bm25.per_term_score(
                    q_term, qtf, exact_match_cnt, dl)
                value_score += score_per_q_term
                n_exact_match += 1
            elif top_k_terms:
                top_term = top_k_terms[0]
                # We use top score only
                multiplier = query_factor(q_term, qtf) * denom_factor
                per_term_entry = TermAlignInfo(q_term, top_term, multiplier)
                per_unknown_tf.append(per_term_entry)

        c_log.debug("Query has %d terms. %d have exact match. %s have align candidates",
                    len(q_counter), n_exact_match, len(per_unknown_tf))

        self.avg_match_count.append(len(per_unknown_tf), 1)
        threshold = 0.1
        if self.avg_match_count.get_average() < threshold:
            c_log.warn("Average alignment was below %f for previous 200 q/d", threshold)

        return K, per_unknown_tf, value_score

