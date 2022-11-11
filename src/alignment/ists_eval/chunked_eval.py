from abc import ABC, abstractmethod
from typing import List

from alignment.ists_eval.eval_utils import save_ists_predictions
from alignment.ists_eval.path_helper import get_ists_save_path
from dataset_specific.ists.parse import iSTSProblemWChunk, AlignmentPrediction
from dataset_specific.ists.path_helper import load_ists_problems_w_chunk
from misc_lib import TEL


class ISTSChunkedSolver(ABC):
    @abstractmethod
    def batch_solve(self, problems: List[iSTSProblemWChunk]) -> List[AlignmentPrediction]:
        pass


# NB: Non batch version.
class ISTSChunkedSolverNB(ISTSChunkedSolver):
    def batch_solve(self, problems: List[iSTSProblemWChunk]) -> List[AlignmentPrediction]:
        output = []
        for t in TEL(problems):
            output.append(self.solve_one(t))
        return output

    @abstractmethod
    def solve_one(self, problem: iSTSProblemWChunk) -> AlignmentPrediction:
        pass


def chunked_solve_and_save_eval(solver: ISTSChunkedSolver,
                                run_name, genre, split):
    problems: List[iSTSProblemWChunk] = load_ists_problems_w_chunk(genre, split)
    ists_predictions: List[AlignmentPrediction] = solver.batch_solve(problems)
    save_path = get_ists_save_path(genre, split, run_name)
    save_ists_predictions(ists_predictions, save_path)


