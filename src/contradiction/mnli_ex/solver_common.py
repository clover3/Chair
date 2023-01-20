from abc import ABC, abstractmethod
from typing import List, Tuple

from attribution.attrib_types import TokenScores
from contradiction.mnli_ex.nli_ex_common import NLIExEntry


class MNLIExSolver(ABC):
    @abstractmethod
    def explain(self, data: List[NLIExEntry], target_label) -> List[Tuple[TokenScores, TokenScores]]:
        pass