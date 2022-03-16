import os

from cpath import output_path
from models.keras_model.bert_keras.v1_load_util import load_model_cls_probe_from_v1_checkpoint


def load_probe(model_config):
    save_path = os.path.join(output_path, "model", "runs", "nli_probe_cls3", "model-100000")
    model, bert_cls_probe = load_model_cls_probe_from_v1_checkpoint(save_path, model_config)
    return model, bert_cls_probe