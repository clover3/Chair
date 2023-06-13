from dataset_specific.scientsbank.pte_solver_if import PTESolverIF
from dataset_specific.scientsbank.pte_data_types import ReferenceAnswer, Facet
from typing import List, Callable

from trainer_v2.keras_server.name_short_cuts import NLIPredictorSig
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.per_project.tli.pte.solver_adapter import get_score_for_facet
from trainer_v2.per_project.tli.token_level_inference import Numpy2D, Numpy1D, TokenLevelInference


# NLI-based
#  1) Make token1 + token2 as a hypothesis
#  2) Judge each tokens are entailed and average the scores

# NLITS-based
#  1) Make [MASK] token1 [MASK] token2 [MASK] as a hypothesis
#  2) Build TLI scoring, and take the score for token1 and token2


class PTESolverTLI(PTESolverIF):
    def __init__(self,
                 nli_predict_fn: NLIPredictorSig,
                 enum_subseq,
                 combine_tli: Callable[[Numpy2D], Numpy1D],
                 name
                 ):
        self.nli_predict_fn = nli_predict_fn
        self.tli_module = TokenLevelInference(nli_predict_fn, enum_subseq)
        self.combine_tli: Callable[[Numpy2D], Numpy1D] = combine_tli
        self.tli_cache = {}
        self.name = name

    def get_name(self):
        return self.name

    def solve(self,
              reference_answer: ReferenceAnswer,
              student_answer: str,
              facet: Facet) -> float:
        premise_like = student_answer
        hypothesis_like = reference_answer.text
        h_tokens: List[str] = [t.text for t in reference_answer.tokens]
        key = premise_like, hypothesis_like
        if key in self.tli_cache:
            tli_output = self.tli_cache[key]
        else:
            tli_output = self.tli_module.do_one(premise_like, h_tokens)
            self.tli_cache[key] = tli_output
        tli_reduce = self.combine_tli(tli_output)
        output_score = get_score_for_facet(reference_answer, facet, tli_reduce)
        return float(output_score)
