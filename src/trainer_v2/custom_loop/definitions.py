import abc
import dataclasses


@dataclasses.dataclass
class ModelConfigType:
    __metaclass__ = abc.ABCMeta
    max_seq_length = abc.abstractproperty()
    num_classes = abc.abstractproperty()


class ModelConfig600_2(ModelConfigType):
    max_seq_length = 600
    num_classes = 2


class ModelConfig300_2(ModelConfigType):
    max_seq_length = 300
    num_classes = 2


class ModelConfig600_3(ModelConfigType):
    max_seq_length = 600
    num_classes = 3
