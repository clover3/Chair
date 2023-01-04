from abc import ABC, abstractmethod


class BioClaimRetrievalSystem(ABC):
    @abstractmethod
    def score(self, question: str, claim: str) -> float:
        pass