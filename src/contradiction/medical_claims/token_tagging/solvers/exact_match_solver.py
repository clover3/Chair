import math
from typing import List, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from models.classic.stopword import load_stopwords


def overlap_check_1_if_no(text1_tokens, text2_tokens) -> List[int]:
    scores = []
    for t1 in text1_tokens:
        if t1 in text2_tokens:
            scores.append(0)
        else:
            scores.append(1)
    return scores


# Assign 1 if no exact match exists
class ExactMatchSolver(TokenScoringSolverIF):
    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = overlap_check_1_if_no(text1_tokens, text2_tokens)
        scores2 = overlap_check_1_if_no(text2_tokens, text1_tokens)
        return scores1, scores2


# Assign 1 if no exact match exists
class ExactMatchSTHandleSolver(TokenScoringSolverIF):
    def __init__(self):
        self.stopwords = load_stopwords()

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.overlap_check_1_if_no(text1_tokens, text2_tokens)
        scores2 = self.overlap_check_1_if_no(text2_tokens, text1_tokens)
        return scores1, scores2

    def overlap_check_1_if_no(self, text1_tokens, text2_tokens) -> List[int]:
        scores = []
        for t1 in text1_tokens:
            if t1 in self.stopwords:
                if t1 in text2_tokens:
                    s = 0.2
                else:
                    s = 1 - 0.2
            else:
                if t1 in text2_tokens:
                    s = 0
                else:
                    s = 1

            scores.append(s)
        return scores


# Assign 1 if no exact match exists
class TF_IDF(TokenScoringSolverIF):
    def __init__(self, df, ctf, stemmer):
        self.ctf = ctf
        self.df = df
        self.stemmer = stemmer

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.overlap_check_1_if_no(text1_tokens, text2_tokens)
        scores2 = self.overlap_check_1_if_no(text2_tokens, text1_tokens)
        return scores1, scores2

    def get_weight(self, token) -> float:
        stemmed_token = self.stemmer(token)
        df = self.df[stemmed_token]
        if df == 0:
            df = 10

        assert self.ctf - df + 0.5 > 0
        return math.log((self.ctf - df + 0.5) / (df + 0.5))

    def overlap_check_1_if_no(self, text1_tokens, text2_tokens) -> List[float]:
        scores = []
        for t1 in text1_tokens:
            weight = self.get_weight(t1)
            if t1 in text2_tokens:
                tf = sum([1 if t1 == t2 else 0 for t2 in text2_tokens])
                assert tf > 0
                s: float = -tf * weight
            else:
                s: float = 1 * weight
            scores.append(s)
        return scores

