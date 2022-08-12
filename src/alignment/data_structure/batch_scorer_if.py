import abc
from typing import List, Tuple, TypeVar

AlignProblem = Tuple[List[str], List[str]]
AlignAnswer = List[List[float]]

class BatchMatrixScorerIF(abc.ABC):
    Problem = AlignProblem
    Answer = AlignAnswer

    @abc.abstractmethod
    def solve(self, problem_list: List[Problem]) -> List[Answer]:
        pass


B = TypeVar('B')
# Batch Align Solver Adapter
class BASAIF(abc.ABC):
    @abc.abstractmethod
    def neural_worker(self, items: List[B]):
        pass

    @abc.abstractmethod
    def reduce(self, t1, t2, output: List) -> List[List[float]]:
        pass

    @abc.abstractmethod
    def enum_child(self, t1: List[str], t2: List[str]) -> List[B]:
        pass