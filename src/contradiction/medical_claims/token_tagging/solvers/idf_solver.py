import math
from typing import List, Tuple

import numpy as np

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF


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
        try:
            stemmed_token = self.stemmer(token)
        except UnicodeDecodeError:
            print("Fail to stem {}".format(token))
            stemmed_token = token
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


class ConvolutionSolver(TokenScoringSolverIF):
    def __init__(self, inner_solver: TokenScoringSolverIF, mask):
        self.inner_solver = inner_solver
        self.mask = mask

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1, scores2 = self.inner_solver.solve(text1_tokens, text2_tokens)
        return masking(scores1, self.mask), masking(scores2, self.mask)


def masking(v: List[float], mask) -> List[float]:
    assert len(mask) % 2 == 1
    w = int(len(mask) / 2)
    a_ex = np.pad(v, [w, w], 'mean')
    c = np.convolve(a_ex, mask, 'valid')
    assert len(v) == len(c)
    return c.tolist()


def main():
    a = [1, 3, 5, 7]
    mask = [0.25, 0.5, 0.25]
    expansion_width = int(len(mask) / 2)
    w = expansion_width
    a_ex = np.pad(a, [w, w], 'mean')
    print(a_ex)
    c = np.convolve(a_ex, mask, 'valid')
    print(c, len(c))


if __name__ == "__main__":
    main()