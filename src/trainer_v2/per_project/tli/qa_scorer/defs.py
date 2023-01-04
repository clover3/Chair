from abc import ABC, abstractmethod


class QAScorer(ABC):
    @abstractmethod
    def score(self, question: str, claim: str) -> float:
        pass