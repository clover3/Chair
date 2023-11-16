from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TokenizedText


class TextRep:
    def __init__(self, tokenized_text: TokenizedText):
        self.tokenized_text = tokenized_text
        self.counter = Counter(self.tokenized_text.sp_tokens)

        indices = {t: list() for t in self.counter}
        for idx, sp_token in enumerate(self.tokenized_text.sp_tokens):
            indices[sp_token].append(idx)
        self.indices: Dict[str, List[int]] = indices

    def get_bow(self):
        for term, cnt in self.counter.items():
            yield term, cnt, self.indices[term]

    def get_terms(self):
        return self.indices.keys()

    @classmethod
    def from_text(cls, tokenizer, text):
        tt = TokenizedText.from_text(tokenizer, text)
        return TextRep(tt)

    def get_sp_size(self):
        return len(self.tokenized_text.sp_tokens)


class BM25T2:
    """
    This scorer references pep_scorer when it scores the pairs.

    - For now it skips stemming
    - If term pair is elite in the given query and document, the score gets boosted
    -
    """
    def __init__(self,
                 pep_scorer: Callable[[List[Tuple[List[str], List[str]]]], List[float]],
                 bm25_scoring_fn: Callable[[int, int, int, int], float],
                 df: Dict[str, int],
                 ):
        self.pep_scorer = pep_scorer
        self.bm25_scoring_fn: Callable[[int, int, int, int], float] = bm25_scoring_fn
        self.tokenizer = get_tokenizer()
        self.q_rep_cache = {}
        self.df = df

    def pep_scorer_wrap(self, q_tokens, d_tokens) -> float:
        def sp_tokens_to_sb_tokens(sp_tokens):
            return list(flatten(map(self.tokenizer.wordpiece_tokenizer.tokenize, sp_tokens)))

        q: List[str] = sp_tokens_to_sb_tokens(q_tokens)
        d: List[str] = sp_tokens_to_sb_tokens(d_tokens)
        score = self.pep_scorer([(q, d)])[0]
        return score

    def check_elite(self, sp_token: str, text_rep: TextRep, is_query: bool):
        """
        It checks if sp_token is elite term in text_rep
        :param sp_token: space tokenized tokens
        :param text_rep:
        :param is_query:
        :return:
        """
        c_log.debug("check_elite(%s, ... )", sp_token)
        def is_gain_large(before, after):
            # TODO add debug msg
            return after - before > 0.5

        base_score: float = self.pep_scorer_wrap([sp_token], [sp_token])
        c_log.debug("base_score=%.2f", base_score)

        for idx in text_rep.indices[sp_token]:
            def enum_candidate_context():
                if 0 <= idx - 1:
                    yield text_rep.tokenized_text.sp_tokens[idx - 1: idx + 1]
                if idx + 1 < len(text_rep.tokenized_text.sp_tokens):
                    yield text_rep.tokenized_text.sp_tokens[idx: idx + 2]

            for tokens in enum_candidate_context():
                if is_query:
                    new_score = self.pep_scorer_wrap(tokens, [sp_token])
                    c_log.debug("pep(%s, %s)=%.2f", str(tokens), str([sp_token]), new_score)
                else:
                    new_score = self.pep_scorer_wrap([sp_token], tokens)
                    c_log.debug("pep(%s, %s)=%.2f", str([sp_token]), str(tokens), new_score)
                if is_gain_large(base_score, new_score):
                    return True
        return False

    def score(self, qd_iter: List[Tuple[str, str]]) -> List[float]:
        output = []
        for q, d in qd_iter:
            f = self.score_one(q, d)
            output.append(f)
        return output

    def score_one(self, query: str, document: str) -> float:
        c_log.debug("score_one()")
        c_log.debug("Query: %s", query)
        c_log.debug("Doc: %s", document)
        if query in self.q_rep_cache:
            q_rep = self.q_rep_cache[query]
        else:
            q_rep = TextRep.from_text(self.tokenizer, query)
            self.q_rep_cache[query] = q_rep

        d_rep = TextRep.from_text(self.tokenizer, document)
        return self.score_from_text_rep(q_rep, d_rep)

    def score_from_text_rep(self, q_rep: TextRep, d_rep: TextRep) -> float:
        def adjust(base_score: float, f_is_q_elite: bool, f_is_d_elite: bool) -> float:
            # TODO find out how much gain is ideal
            score = base_score
            if f_is_q_elite:
                score = score * 1.2

            if f_is_d_elite:
                score = score * 1.2

            return score

        score_sum = 0
        dl = d_rep.get_sp_size()
        for q_term in q_rep.get_terms():
            f_is_q_elite: bool = self.check_elite(q_term, q_rep, is_query=True)

            raw_tf = d_rep.counter[q_term]
            qf = q_rep.counter[q_term]
            qdf = self.df[q_term] if q_term in self.df else 0
            base_score = self.bm25_scoring_fn(raw_tf, qf, dl, qdf)

            if raw_tf > 0:
                f_is_d_elite = self.check_elite(q_term, d_rep, is_query=False)
            else:
                f_is_d_elite = False

            # This is equal to the base_score if not elite
            boosted_score = adjust(base_score, f_is_q_elite, f_is_d_elite)

            score_sum += boosted_score

        _ = input("press something to continue")
        return score_sum

