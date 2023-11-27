import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TextRep


def get_term_pairs_for_elite_check(sp_token: str, text_rep: TextRep, is_query: bool) -> List[Tuple[List[str], List[str]]]:
    base_input: List[Tuple[List[str], List[str]]] = [([sp_token], [sp_token]), ]

    context_to_check: List[List[str]] = []
    for idx in text_rep.indices[sp_token]:
        def enum_candidate_context() -> Iterable[List[str]]:
            if 0 <= idx - 1:
                yield text_rep.tokenized_text.sp_tokens[idx - 1: idx + 1]
            if idx + 1 < len(text_rep.tokenized_text.sp_tokens):
                yield text_rep.tokenized_text.sp_tokens[idx: idx + 2]

        for tokens in enum_candidate_context():
            context_to_check.append(tokens)

    context_to_check_paired: List[Tuple[List[str], List[str]]] = []
    for tokens in context_to_check:
        if is_query:
            paired = (tokens, [sp_token])
        else:
            paired = ([sp_token], tokens)
        context_to_check_paired.append(paired)

    return base_input + context_to_check_paired


class BM25T2:
    """
    This scorer references pep_scorer when it scores the pairs.

    - For now it skips stemming
    - If term pair is elite in the given query and document, the score gets boosted
    - Ideally, we would precompute elite score patterns and apply them in the inference time.
    - As a proof of concept, we run pep on the fly so that we don't miss any "elite indicating contexts"
        - It is possible that it includes many disambiguation contexts other than the elite indicating contexts.
        - If the term is topic independent, it is more likely to be "elite indicating contexts"
    -
    """
    def __init__(self,
                 term_pair_scorer: Callable[[List[Tuple[str, str]]], List[float]],
                 bm25_scoring_fn: Callable[[int, int, int, int], float],
                 df: Dict[str, int],
                 ):
        self.term_pair_scorer = term_pair_scorer
        self.bm25_scoring_fn: Callable[[int, int, int, int], float] = bm25_scoring_fn
        self.tokenizer = get_tokenizer()
        self.q_rep_cache = {}
        self.df = df
        self.is_elite_cache = {}
        self.logged = set()

    def get_q_rep(self, query):
        if query in self.q_rep_cache:
            q_rep = self.q_rep_cache[query]
        else:
            q_rep = TextRep.from_text(self.tokenizer, query)
            self.q_rep_cache[query] = q_rep
        return q_rep

    def _term_scorer_batch(self, qd_list: List[Tuple[List[str], List[str]]]) -> List[float]:
        # qd_list are space tokenized
        def sp_tokens_to_sb_tokens(sp_tokens):
            return list(flatten(map(self.tokenizer.wordpiece_tokenizer.tokenize, sp_tokens)))

        def reform_pair(qd):
            q_sp, d_sp = qd
            return " ".join(q_sp), " ".join(d_sp)
        payload = list(map(reform_pair, qd_list))
        scores = self.term_pair_scorer(payload)
        return scores

    def _check_elite_w_cache(self, sp_token: str, text_rep: TextRep, is_query: bool):
        if is_query:
            key: Tuple[str, str] = " ".join(text_rep.tokenized_text.sp_tokens), sp_token
            if key in self.is_elite_cache:
                return self.is_elite_cache[key]
            else:
                f = self._check_elite(sp_token, text_rep, is_query)
                self.is_elite_cache[key] = f
        else:
            return self._check_elite(sp_token, text_rep, is_query)

    def _check_elite(self, sp_token: str, text_rep: TextRep, is_query: bool):
        # c_log.debug("check_elite(%s, ... )", sp_token)
        base_input = [([sp_token], [sp_token]), ]
        base_score: float = self._term_scorer_batch(base_input)[0]
        # c_log.debug("base_score=%.2f", base_score)

        def is_gain_large(after):
            return after - base_score > 0.5 and after > 2

        context_to_check: List[List[str]] = []
        for idx in text_rep.indices[sp_token]:
            def enum_candidate_context() -> Iterable[List[str]]:
                if 0 <= idx - 1:
                    yield text_rep.tokenized_text.sp_tokens[idx - 1: idx + 1]
                if idx + 1 < len(text_rep.tokenized_text.sp_tokens):
                    yield text_rep.tokenized_text.sp_tokens[idx: idx + 2]

            for tokens in enum_candidate_context():
                context_to_check.append(tokens)

        context_to_check_paired: List[Tuple[List[str], List[str]]] = []
        for tokens in context_to_check:
            if is_query:
                paired = (tokens, [sp_token])
            else:
                paired = ([sp_token], tokens)
            context_to_check_paired.append(paired)

        scores = self._term_scorer_batch(context_to_check_paired)
        max_idx = np.argmax(scores)

        q_tokens, d_tokens = context_to_check_paired[max_idx]
        best_context = " ".join(q_tokens) + " [SEP] " + " ".join(d_tokens)
        if best_context not in self.logged :
            c_log.debug("Best context: %s %.2f -> %.2f (%s)", best_context, base_score, scores[max_idx], str(is_gain_large(scores[max_idx])))
            self.logged.add(best_context)
        is_elite_arr = list(map(is_gain_large, scores))
        is_elite: bool = any(is_elite_arr)
        return is_elite

    def score_from_text_rep(self, q_rep: TextRep, d_rep: TextRep) -> float:
        c_log.debug("Query: %s", " ".join(q_rep.tokenized_text.sp_tokens))
        c_log.debug("Doc: %s", " ".join(d_rep.tokenized_text.sp_tokens))
        def adjust(base_score: float, f_is_q_elite: bool, f_is_d_elite: bool) -> float:
            score = base_score
            if f_is_q_elite:
                score = score * 1.2

            if f_is_d_elite:
                score = score * 1.2

            return score

        score_sum = 0
        dl = d_rep.get_sp_size()
        for q_term in q_rep.get_terms():
            f_is_q_elite: bool = self._check_elite_w_cache(q_term, q_rep, is_query=True)
            raw_tf = d_rep.counter[q_term]
            qf = q_rep.counter[q_term]
            qdf = self.df[q_term] if q_term in self.df else 0
            base_score = self.bm25_scoring_fn(raw_tf, qf, dl, qdf)

            f_is_d_elite = raw_tf > 0 and self._check_elite(q_term, d_rep, is_query=False)
            boosted_score = adjust(base_score, f_is_q_elite, f_is_d_elite)
            score_sum += boosted_score

        return score_sum

    def score_with_elite_info(self, q_rep: TextRep, d_rep: TextRep, elite_info: List[Tuple[bool, bool]]):
        def adjust(base_score: float, f_is_q_elite: bool, f_is_d_elite: bool) -> float:
            score = base_score
            if f_is_q_elite:
                score = score * 1.2

            if f_is_d_elite:
                score = score * 1.2

            return score

        score_sum = 0
        dl = d_rep.get_sp_size()
        for idx, q_term in enumerate(q_rep.get_terms()):
            q_elite, d_elite = elite_info[idx]
            raw_tf = d_rep.counter[q_term]
            qf = q_rep.counter[q_term]
            qdf = self.df[q_term] if q_term in self.df else 0
            base_score = self.bm25_scoring_fn(raw_tf, qf, dl, qdf)
            boosted_score = adjust(base_score, q_elite, d_elite)
            score_sum += boosted_score
        return score_sum

    def get_elite_info(self, qd: Tuple[TextRep, TextRep]) -> List[Tuple[bool, bool]]:
        q_rep, d_rep = qd
        output = []
        for q_term in q_rep.get_terms():
            f_is_q_elite: bool = self._check_elite(q_term, q_rep, is_query=True)
            raw_tf = d_rep.counter[q_term]
            f_is_d_elite = raw_tf > 0 and self._check_elite(q_term, d_rep, is_query=False)
            output.append((f_is_q_elite, f_is_d_elite))
        return output

    def score(self, qd_iter: Iterable[Tuple[str, str]]) -> List[float]:
        def convert_to_reps(qd):
            q, d = qd
            q_rep = self.get_q_rep(q)
            d_rep = TextRep.from_text(self.tokenizer, d)
            return q_rep, d_rep

        qd_reps: List[Tuple[TextRep, TextRep]] = list(map(convert_to_reps, qd_iter))
        output: List[float] = [self.score_from_text_rep(q, d) for q, d in qd_reps]
        return output