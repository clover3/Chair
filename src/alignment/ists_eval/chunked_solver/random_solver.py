import random

from alignment.ists_eval.chunked_eval import ISTSChunkedSolverNB
from alignment.ists_eval.prediction_helper import get_alignment_label_units
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction, ALIGN_EQUI


class RandomSolver(ISTSChunkedSolverNB):
    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        n_chunk1 = len(problem.chunks1)
        n_chunk2 = len(problem.chunks2)

        indices1 = list(range(n_chunk1))
        indices2 = list(range(n_chunk2))
        random.shuffle(indices1)
        random.shuffle(indices2)

        labels = [[ALIGN_EQUI] for _ in range(min(n_chunk1, n_chunk2))]

        return get_alignment_label_units(indices1, indices2, labels, problem)


class LocationSolver(ISTSChunkedSolverNB):
    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        n_chunk1 = len(problem.chunks1)
        n_chunk2 = len(problem.chunks2)
        indices1 = list(range(n_chunk1))
        indices2 = list(range(n_chunk2))
        labels = [[ALIGN_EQUI] for _ in range(min(n_chunk1, n_chunk2))]
        return get_alignment_label_units(indices1, indices2, labels, problem)
