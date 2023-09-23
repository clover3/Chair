from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class AttentionExtractor(ABC):
    @abstractmethod
    def predict_list(self, tokens_pair_list: List[Tuple[List, List]]) -> List[np.array]:
        pass