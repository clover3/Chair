from abc import ABC, abstractmethod
from typing import NamedTuple, List


class ContProblem(NamedTuple):
    question: str
    claim1_text: str
    claim2_text: str
    label: int

    def signature(self):
        return "{}_{}_{}".format(self.question, self.claim1_text, self.claim2_text)

    @classmethod
    def from_json(cls, j):
        return ContProblem(j['question'], j['claim1_text'], j['claim2_text'], j['label'])


class ContClassificationSolverIF(ABC):
    @abstractmethod
    def solve_batch(self, problems: List[ContProblem]) -> List[int]:
        pass


class ContClassificationProbabilityScorer(ABC):
    @abstractmethod
    def solve_batch(self, problems: List[ContProblem]) -> List[float]:
        pass


class ContClassificationSolverNB(ContClassificationSolverIF):
    def solve_batch(self, problems: List[ContProblem]) -> List[int]:
        return [self.solve(p) for p in problems]

    @abstractmethod
    def solve(self, problem: ContProblem) -> int:
        pass


NOTE_POS_PAIR = "pos pair"
NOTE_NEG_TYPE1_YS = "neg from same group yes"
NOTE_NEG_TYPE1_NO = "neg from same group no"
NOTE_NEG_TYPE2 = "neg from different group"