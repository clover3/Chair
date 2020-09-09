import abc
from abc import ABC
from typing import Dict, NamedTuple, List


class CPPNCGeneratorInterface(ABC):
    @abc.abstractmethod
    def generate_instances(self, claim: Dict, data_id_manager):
        pass


class PayloadAsTokens(NamedTuple):
    passage: List[str]
    text1: List[str]
    text2: List[str]
    data_id: int
    is_correct: int