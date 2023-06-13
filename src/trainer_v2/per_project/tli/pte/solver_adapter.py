from typing import Callable, List

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF, \
    TokenScoringSolverIFOneWay
from dataset_specific.scientsbank.pte_solver_if import PTESolverIF
from dataset_specific.scientsbank.pte_data_types import ReferenceAnswer, Facet


def get_score_for_facet(
        reference_answer: ReferenceAnswer,
        facet: Facet,
        score_array)  -> float:
    idx1, idx2 = reference_answer.facet_location(facet)
    if idx1 is None:
        output_score = score_array[idx2]
    elif idx2 is None:
        output_score = score_array[idx1]
    else:
        output_score = (score_array[idx1] + score_array[idx2]) / 2
    return float(output_score)


class PTESolverFromTokenScoringSolver(PTESolverIF):
    def __init__(self,
                 t_solver: TokenScoringSolverIF,
                 sp_tokenizer: Callable[[str], List[str]],
                 reverse_score: bool,
                 name: str):
        self.t_solver = t_solver
        self.score_array_d = {}
        self.name = name
        self.sp_tokenizer = sp_tokenizer
        self.reverse_score = reverse_score

    def subtract_from_one(self, score_array):
        return [1 - t for t in score_array]

    def solve(self,
              reference_answer: ReferenceAnswer,
              student_answer: str,
              facet: Facet) -> float:
        premise_like = student_answer
        hypothesis_like = reference_answer.text
        h_tokens: List[str] = [t.text for t in reference_answer.tokens]
        key = premise_like, hypothesis_like
        # We assume that given prem/hypo, the h_tokens are constant

        if key in self.score_array_d:
            score_array = self.score_array_d[key]
        else:
            if isinstance(self.t_solver, TokenScoringSolverIFOneWay):
                p_tokens = self.sp_tokenizer(premise_like)
                score_array = self.t_solver.solve_one_way(p_tokens, h_tokens)
            else:
                p_tokens = self.sp_tokenizer(premise_like)
                _, score_array = self.t_solver.solve(p_tokens, h_tokens)

            if self.reverse_score:
                score_array = self.subtract_from_one(score_array)

        output_score = get_score_for_facet(reference_answer, facet, score_array)
        return float(output_score)

    def get_name(self):
        return self.name
