import abc
import dataclasses


@dataclasses.dataclass
class ModelConfigType:
    __metaclass__ = abc.ABCMeta
    max_seq_length = abc.abstractproperty()
    num_classes = abc.abstractproperty()


@dataclasses.dataclass
class HFModelConfigType(ModelConfigType):
    __metaclass__ = abc.ABCMeta
    max_seq_length = abc.abstractproperty()
    num_classes = abc.abstractproperty()
    model_type = abc.abstractproperty()


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


class ModelConfig256_1(HFModelConfigType):
    max_seq_length = 256
    num_classes = 1
    model_type = "bert-base-uncased"



class ModelConfig256_2(HFModelConfigType):
    max_seq_length = 256
    num_classes = 2
    model_type = "bert-base-uncased"


class ModelConfig512_1(HFModelConfigType):
    max_seq_length = 512
    num_classes = 1
    model_type = "bert-base-uncased"


class ModelConfig512_2(HFModelConfigType):
    max_seq_length = 512
    num_classes = 2
    model_type = "bert-base-uncased"


class ModelConfig2Seg:
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3