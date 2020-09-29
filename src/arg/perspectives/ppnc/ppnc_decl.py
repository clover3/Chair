import abc
from abc import ABC
from typing import Dict


class CPPNCGeneratorInterface(ABC):
    @abc.abstractmethod
    def generate_instances(self, claim: Dict, data_id_manager):
        pass


