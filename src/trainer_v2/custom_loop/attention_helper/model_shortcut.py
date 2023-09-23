from cpath import get_canonical_model_path2
from trainer_v2.custom_loop.attention_helper.attention_extractor import AttentionExtractor1
from trainer_v2.custom_loop.attention_helper.attention_extractor_interface import AttentionExtractor
from trainer_v2.custom_loop.modeling_common.bert_common import ModelConfig300_3


def load_nli14_attention_extractor():
    model_path = nli14_model_path()
    model_config = ModelConfig300_3()
    ae = AttentionExtractor1(model_path, model_config)
    return ae


def nli14_model_path():
    model_path = get_canonical_model_path2("nli14_0", "model_12500")
    return model_path


def nlits87_model_path():
    model_path = get_canonical_model_path2("nli_ts_run87_0", "model_12500")
    return model_path
