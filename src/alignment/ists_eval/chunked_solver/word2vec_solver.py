from typing import List

import gensim

from alignment.ists_eval.chunked_eval import ISTSChunkedSolverNB
from alignment.ists_eval.chunked_solver.solver_common import get_similarity_table
from alignment.ists_eval.prediction_helper import score_matrix_to_alignment_by_rank
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction
from misc_lib import average
from trainer_v2.chair_logging import c_log


def w2v_sim(w2v, word1, word2) -> float:
    try:
        s = w2v.similarity(word1, word2)
        return s
    except KeyError as e:
        if word1 == word2:
            return 1
        return 0


class Word2VecChunkHelper:
    def __init__(self, word2vec_path):
        c_log.info("Loading word2vec...")
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        c_log.info("Done")

    def similar(self, word1, word2) -> float:
        return w2v_sim(self.w2v, word1, word2)

    def score_chunk_pair(self, chunk1, chunk2) -> float:
        def tokenize_normalize(chunk):
            tokens = chunk.lower().split()
            return tokens

        tokens1 = tokenize_normalize(chunk1)
        tokens2 = tokenize_normalize(chunk2)

        if len(tokens2) > len(tokens1):
            tmp = tokens2
            tokens2 = tokens1
            tokens1 = tmp

        scores = self.get_max_similarity(tokens1, tokens2)
        return average(scores)

    def get_max_similarity(self, text1_tokens, text2_tokens) -> List[float]:
        scores = []
        for t1 in text1_tokens:
            similarity_list: List[float] = [self.similar(t1, t2) for t2 in text2_tokens]
            scores.append(max(similarity_list))

        return scores


class Word2VecSolver(ISTSChunkedSolverNB):
    def __init__(self, word2vec_path):
        self.chunk_helper = Word2VecChunkHelper(word2vec_path)

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        table = get_similarity_table(problem, self.chunk_helper.score_chunk_pair)
        return score_matrix_to_alignment_by_rank(table, problem)

