import abc
import dataclasses


@dataclasses.dataclass
class ModelConfigType:
    __metaclass__ = abc.ABCMeta
    max_seq_length = abc.abstractproperty()
    num_classes = abc.abstractproperty()
