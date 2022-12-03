from alignment.ists_eval.chunked_eval import ISTSChunkedSolverNB
from alignment.ists_eval.chunked_solver.solver_common import get_similarity_table
from alignment.ists_eval.prediction_helper import score_matrix_to_alignment_by_rank
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction


def score_chunk_pair_exact_match(chunk1, chunk2) -> float:
    def tokenize_normalize(chunk):
        tokens = chunk.lower().split()
        return tokens

    tokens1 = tokenize_normalize(chunk1)
    tokens2 = tokenize_normalize(chunk2)

    if len(tokens2) > len(tokens1):
        tmp = tokens2
        tokens2 = tokens1
        tokens1 = tmp

    cnt = 0
    for t in tokens1:
        if t in tokens2:
            cnt += 1
    return cnt / len(tokens1) if len(tokens1) else 0


class ExactMatchSolver(ISTSChunkedSolverNB):
    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:

        table = get_similarity_table(problem, score_chunk_pair_exact_match)
        return score_matrix_to_alignment_by_rank(table, problem)

