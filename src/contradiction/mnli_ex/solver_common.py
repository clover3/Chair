from abc import ABC, abstractmethod
from typing import List, Tuple

from attribution.attrib_types import TokenScores
from contradiction.mnli_ex.load_mnli_ex_data import MNLIExEntry


class MNLIExSolver(ABC):
    @abstractmethod
    def explain(self, data: List[MNLIExEntry], target_label) -> List[Tuple[TokenScores, TokenScores]]:
        pass