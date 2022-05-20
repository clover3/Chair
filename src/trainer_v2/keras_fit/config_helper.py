from official.nlp.bert import configs as bert_configs

from cpath import get_bert_config_path


class ModelConfig:
    num_classes = 3
    max_seq_length = 512

    def __init__(self, bert_config, max_seq_length):
        self.bert_config = bert_config
        self.max_seq_length = max_seq_length


class MultiSegModelConfig:
    num_classes = 3
    def __init__(self, bert_config, max_seq_length_list):
        self.bert_config = bert_config
        self.max_seq_length_list = max_seq_length_list


def get_model_config_nli():
    bert_config = get_bert_config()
    max_seq_length = 300
    model_config = ModelConfig(bert_config, max_seq_length)
    return model_config


def get_bert_config():
    bert_config = bert_configs.BertConfig.from_json_file(get_bert_config_path())
    return bert_config