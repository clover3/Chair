from typing import List

import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs


def get_dataset_factory_two_seg(model_config: ModelConfigType):
    max_seq_length = model_config.max_seq_length
    def factory(payload: List):
        def generator():
            for item in payload:
                yield tuple(item)

        int_list = tf.TensorSpec(shape=(max_seq_length,), dtype=tf.int32)
        output_signature = (int_list, int_list)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return dataset
    return factory


def get_local_decision_layer_from_model_by_shape(model, n_label=3):
    for idx, layer in enumerate(model.layers):
        try:
            shape = layer.output.shape
            if shape[1] == 2 and shape[2] == n_label:
                c_log.debug("Maybe this is local decision layer: {}".format(layer.name))
                return layer
        except AttributeError:
            print("layer is actually : ", layer)
        except IndexError:
            pass

    c_log.error("Layer not found")
    for idx, layer in enumerate(model.layers):
        c_log.error(idx, layer, layer.output.shape)
    raise KeyError


def load_local_decision_model(model_config: ModelConfigType, model_path: str):
    model = load_model_by_dir_or_abs(model_path)
    local_decision_layer = get_local_decision_layer_from_model_by_shape(model, model_config.num_classes)
    new_outputs = [local_decision_layer.output, model.outputs]
    model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return model

