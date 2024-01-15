from abc import ABC, abstractmethod
from typing import List, Tuple


class RetrieverIF(ABC):
    @abstractmethod
    def retrieve(self, query, max_item: int) -> List[Tuple[str, float]]:
        pass
