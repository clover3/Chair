import abc
from typing import List


class MMDGenI(abc.ABC):
    @abc.abstractmethod
    def generate(self, data_id_manager, qids):
        pass

    @abc.abstractmethod
    def write(self, insts: List, out_path: str):
        pass
