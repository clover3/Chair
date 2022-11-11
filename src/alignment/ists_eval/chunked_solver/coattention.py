from alignment.ists_eval.chunked_eval import ISTSChunkedSolverNB
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_rank
from alignment.matrix_scorers2.methods.coattention_solver import CoAttentionSolver
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction
from explain.bert_components.attention_extractor import AttentionExtractor


class CoAttentionSolverChunked(ISTSChunkedSolverNB):
    def __init__(self, ae):
        self.inner = CoAttentionSolver(ae)

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        table = self.inner.solve(problem.chunks1, problem.chunks2)
        return score_matrix_to_alignment_by_rank(table, problem)


def get_co_attention_chunked_solver():
    ae = AttentionExtractor()
    return CoAttentionSolverChunked(ae.predict)

