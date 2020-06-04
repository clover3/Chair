from abc import ABC, abstractmethod
from typing import Dict


class QueryWeightGenInterface(ABC):
    def __init__(self):
        self.cache = {}
        self.cache_by_id = {}

    def gen(self, text) -> Dict[str, float]:
        if text in self.cache:
            return self.cache[text]

        r = self.gen_inner(text)
        self.cache[text] = r
        return r

    def gen_w_id(self, text, text_id) -> Dict[str, float]:
        if text_id in self.cache_by_id:
            return self.cache_by_id[text_id]

        r = self.gen_inner(text)
        self.cache_by_id[text_id] = r
        return r

    @abstractmethod
    def gen_inner(self, text) -> Dict[str, float]:
        pass
