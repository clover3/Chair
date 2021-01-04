import abc
from collections import OrderedDict
from typing import Iterable, Any

from arg.qck.decl import QKUnit
from misc_lib import DataIDManager


class InstanceGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, kc_candidate: Iterable[QKUnit],
                       data_id_manager: DataIDManager):
        pass

    def encode_fn(self, any: Any) -> OrderedDict:
        pass