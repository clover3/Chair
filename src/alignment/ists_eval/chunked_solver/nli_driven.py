from alignment.ists_eval.chunked_eval import ISTSChunkedSolver, ISTSChunkedSolverNB
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction, ALIGN_EQUI, ALIGN_SPE1, ALIGN_SPE2, \
    AlignmentLabelUnit, ALIGN_SIMI
from typing import List
from list_lib import lmap
from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig


# Option 1: Work on aligned outputs of Exact match or word2vec
class NLIDrivenSolver(ISTSChunkedSolver):
    def __init__(self, predict_fn: NLIPredictorSig, base_solver: ISTSChunkedSolverNB):
        self.predict_fn = predict_fn
        self.base_solver = base_solver

    def batch_solve(self, problems: List[iSTSProblemWChunk]) -> List[AlignmentPrediction]:
        return lmap(self.solve_one, problems)

    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        base_alignment: AlignmentPrediction = self.base_solver.solve_one(problem)
        problem_id, alignment_list = base_alignment

        def is_entailment(probs):
            return probs[0] > 0.5

        def convert_single_alignment(alignment):
            def predict_one(t1: str, t2: str):
                return self.predict_fn([(t1, t2)])[0]
            nli_pred1: List[float] = predict_one(alignment.chunk_text1, alignment.chunk_text2)
            nli_pred2: List[float] = predict_one(alignment.chunk_text2, alignment.chunk_text1)

            if is_entailment(nli_pred1) and is_entailment(nli_pred2):
                align_types = [ALIGN_EQUI]
            elif is_entailment(nli_pred1) and not is_entailment(nli_pred2):
                align_types = [ALIGN_SPE1]
            elif not is_entailment(nli_pred1) and is_entailment(nli_pred2):
                align_types = [ALIGN_SPE2]
            else:
                align_types = [ALIGN_SIMI]

            return AlignmentLabelUnit(
                alignment.chunk_token_id1,
                alignment.chunk_token_id2,
                alignment.chunk_text1,
                alignment.chunk_text2,
                align_types,
                alignment.align_score
            )

        return problem_id, lmap(convert_single_alignment, alignment_list)
