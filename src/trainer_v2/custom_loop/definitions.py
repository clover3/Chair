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


class ModelConfig300_3(ModelConfigType):
    max_seq_length = 300
    num_classes = 3


class ModelConfig150_3(ModelConfigType):
    max_seq_length = 150
    num_classes = 3


class ModelConfig600_3(ModelConfigType):
    max_seq_length = 600
    num_classes = 3


class ModelConfig2Seg:
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3