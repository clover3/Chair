from trainer_v2.custom_loop.definitions import ModelConfigType


class ModelConfig(ModelConfigType):
    max_seq_length = 300
    num_classes = 3