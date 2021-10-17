import os
import sys

from cpath import output_path
from explain.bert_components.nli300 import ModelConfig
from models.keras_model.bert_keras.v1_load_util import load_model_from_v1_checkpoint


def save_to_v2_model():
    save_path = sys.argv[1]
    model, bert_classifier_layer = load_model_from_v1_checkpoint(save_path, ModelConfig())
    #
    # save_path = os.path.join(output_path, "model", "runs", "standard_nli_v2_modular")
    # model.save(save_path)

    save_path = os.path.join(output_path, "model", "runs", "standard_nli_v2_weights")

    layer_names = set()
    for layer in model.layers:
        print(layer.name)
        if layer.name in layer_names:
            raise KeyError()
        layer_names.add(layer.name)
    model.save_weights(save_path, save_format="h5")


if __name__ == "__main__":
    save_to_v2_model()
