from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple

from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import merge_lms
from arg.qck.token_scoring.decl import ScoreVector, TokenScore
from list_lib import right
from models.classic.lm_util import get_log_odd2


class ScorerInterface(ABC):
    @abstractmethod
    def score_text(self, q_id, text) -> ScoreVector:
        pass

    @abstractmethod
    def score_token(self, q_id, token) -> TokenScore:
        pass

    @abstractmethod
    def zero_score(self) -> TokenScore:
        pass


class LogOddScorer(ScorerInterface):
    def __init__(self, q_lms: List[Tuple[str, Counter]], alpha=0.1):
        bg_lm = merge_lms(right(q_lms))
        self.tokenizer = PCTokenizer()
        print("Eval log odds")
        self.claim_log_odds_dict = {qid: get_log_odd2(q_lm, bg_lm, alpha)
                                    for qid, q_lm in q_lms}

    def score_text(self, q_id, text) -> ScoreVector:
        tokens = self.tokenizer.tokenize_stem(text)
        c_lm = self.claim_log_odds_dict[q_id]
        score = sum([[c_lm[t]] for t in tokens])
        return score

    def score_token(self, q_id, token) -> TokenScore:
        stemmed_token = self.tokenizer.stemmer.stem(token)
        c_lm = self.claim_log_odds_dict[q_id]
        return [c_lm[stemmed_token]]

    def zero_score(self) -> TokenScore:
        return [0.0]


class RawProbabilityScorer(ScorerInterface):
    def __init__(self, q_lms: List[Tuple[str, Counter]]):
        self.q_lm_dict = dict(q_lms)
        self.bg_lm = merge_lms(right(q_lms))
        self.tokenizer = PCTokenizer()

    def score_text(self, q_id, text) -> ScoreVector:
        tokens = self.tokenizer.tokenize_stem(text)
        c_lm = self.q_lm_dict[q_id]

        return list([[c_lm[t], self.bg_lm[t]] for t in tokens])

    def score_token(self, q_id, token) -> TokenScore:
        stemmed_token = self.tokenizer.stemmer.stem(token)
        c_lm = self.q_lm_dict[q_id]
        return [c_lm[stemmed_token], self.bg_lm[stemmed_token]]

    def zero_score(self) -> TokenScore:
        return [0.0, 0.0]
