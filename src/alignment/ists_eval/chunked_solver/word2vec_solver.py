import os
from typing import List, Tuple

import gensim

from alignment.ists_eval.chunked_eval import ISTSChunkedSolverNB
from alignment.ists_eval.chunked_solver.solver_common import get_similarity_table
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_rank
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction
from misc_lib import average


class Word2VecSolver(ISTSChunkedSolverNB):
    def __init__(self, word2vec_path):
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def get_max_similarity(self, text1_tokens, text2_tokens) -> List[float]:
        scores = []
        for t1 in text1_tokens:
            similarity_list: List[float] = [self.similar(t1, t2) for t2 in text2_tokens]
            scores.append(max(similarity_list))

        return scores

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        def tokenize_normalize(chunk):
            tokens = chunk.lower().split()
            return tokens

        def score_chunk_pair(chunk1, chunk2) -> float:
            tokens1 = tokenize_normalize(chunk1)
            tokens2 = tokenize_normalize(chunk2)

            if len(tokens2) > len(tokens1):
                tmp = tokens2
                tokens2 = tokens1
                tokens1 = tmp

            scores = self.get_max_similarity(tokens1, tokens2)
            return average(scores)

        table = get_similarity_table(problem, score_chunk_pair)
        return score_matrix_to_alignment_by_rank(table, problem)

    def similar(self, word1, word2) -> float:
        try:
            s = self.w2v.similarity(word1, word2)
            # print("{} {} {}".format(s, word1, word2))
            return s
        except KeyError:
            msg = "KeyError {} {}".format(word1, word2)
            if not self.w2v.__contains__(word1):
                msg += " missing {}".format(word1)
            if not self.w2v.__contains__(word2):
                msg += " missing {}".format(word2)
            if word1 == word2:
                return 1
            # print(msg)
            return 0
