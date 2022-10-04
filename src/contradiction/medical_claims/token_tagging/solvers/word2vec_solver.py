import os
from typing import List, Tuple

import gensim

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.solvers.ensemble_solver import EnsembleSolver
from contradiction.medical_claims.token_tagging.solvers.exact_match_solver import ExactMatchSolver


class Word2VecSolver(TokenScoringSolverIF):
    def __init__(self, word2vec_path):
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.get_max_similarity_rev(text1_tokens, text2_tokens)
        scores2 = self.get_max_similarity_rev(text2_tokens, text1_tokens)
        return scores1, scores2

    def get_max_similarity(self, text1_tokens, text2_tokens) -> List[float]:
        scores = []
        for t1 in text1_tokens:
            similarity_list: List[float] = [self.similar(t1, t2) for t2 in text2_tokens]
            scores.append(max(similarity_list))

        return scores

    def similar(self, word1, word2) -> float:
        try:
            s = self.w2v.similarity(word1, word2)
            print("{} {} {}".format(s, word1, word2))
            return s
        except KeyError:
            msg = "KeyError {} {}".format(word1, word2)
            if not self.w2v.__contains__(word1):
                msg += " missing {}".format(word1)
            if not self.w2v.__contains__(word2):
                msg += " missing {}".format(word2)
            if word1 == word2:
                return 1
            return 0

    def get_max_similarity_rev(self, text1_tokens, text2_tokens) -> List[float]:
        scores = self.get_max_similarity(text1_tokens, text2_tokens)
        return [1-s for s in scores]


def get_word2vec_solver() -> Word2VecSolver:
    word2vec_path = os.path.join("D:\\data\\embeddings\\GoogleNews-vectors-negative300.bin")
    return Word2VecSolver(word2vec_path)


def get_word2vec_em_solver() -> TokenScoringSolverIF:
    word2vec_path = get_word2vec_path()
    solver_list = [Word2VecSolver(word2vec_path), ExactMatchSolver()]
    return EnsembleSolver(solver_list)


def get_word2vec_path():
    word2vec_path = os.path.join("D:\\data\\embeddings\\GoogleNews-vectors-negative300.bin")
    return word2vec_path


def get_pubmed_word2vec_solver() -> Word2VecSolver:
    word2vec_path = os.path.join("D:\\data\\\chair_output\\w2v\\BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    return Word2VecSolver(word2vec_path)


class W2VAntonymSolver(TokenScoringSolverIF):
    def __init__(self, word2vec_path):
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.get_best_score(text1_tokens, text2_tokens)
        scores2 = self.get_best_score(text2_tokens, text1_tokens)
        return scores1, scores2

    def get_best_score(self, text1_tokens, text2_tokens) -> List[float]:
        scores = []
        for t1 in text1_tokens:
            likely_per_token: List[float] = [self.likely(t1, t2) for t2 in text2_tokens]
            scores.append(max(likely_per_token))

        return scores

    def likely(self, word1, word2) -> float:
        try:
            s = self.w2v.similarity(word1, word2)
            if word1 == word2:
                return 0
            else:
                return s
        except KeyError:
            return 0


def get_w2v_antonym():
    word2vec_path = get_word2vec_path()
    return W2VAntonymSolver(word2vec_path)