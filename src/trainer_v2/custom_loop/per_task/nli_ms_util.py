from typing import List

import tensorflow as tf

from arg.qck.encode_common import encode_single
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import ceil_divide
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.per_task.nli_ts_util import batch_shaping, EncodedSegmentIF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config


def get_local_decision_layer_from_model_by_shape(model):
    for idx, layer in enumerate(model.layers):
        try:
            shape = layer.output.shape
            if shape[3] == 3:
                print("Maybe this is local decision layer: {}".format(layer.name))
                return layer
        except AttributeError:
            print("layer is actually : ", layer)
        except IndexError:
            pass

    print("Layer not found")
    for idx, layer in enumerate(model.layers):
        print(idx, layer, layer.output.shape)
    raise KeyError


def get_weight_layer_from_model_by_shape(model):
    for idx, layer in enumerate(model.layers):
        try:
            shape = layer.output.shape
            if shape[3] == 1:
                print("Maybe this is local decision layer: {}".format(layer.name))
                return layer
        except AttributeError:
            print("layer is actually : ", layer)
        except IndexError:
            pass

    print("Layer not found")
    for idx, layer in enumerate(model.layers):
        print(idx, layer, layer.output.shape)
    raise KeyError


class KerasPredictHelper:
    def __init__(self, model, strategy, data_reform_fn):
        self.strategy = strategy
        self.model = model
        self.n_input = len(self.model.inputs)
        self.data_reform_fn = data_reform_fn

        def get_spec(input_i: tf.keras.layers.Input):
            return tf.TensorSpec(input_i.type_spec.shape[1:], dtype=tf.int32)
        self.out_sig = [get_spec(input) for input in model.inputs]

    def predict(self, input_list):
        batch_size = 16
        while len(input_list) % batch_size:
            input_list.append(input_list[-1])

        def generator():
            yield from input_list

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tuple(self.out_sig))
        strategy = self.strategy

        dataset = dataset.map(self.data_reform_fn)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        maybe_step = ceil_divide(len(input_list), batch_size)
        dataset = distribute_dataset(strategy, dataset)
        model = self.model
        return model.predict(dataset, steps=maybe_step)



class LocalDecisionNLIMS:
    def __init__(self, model, strategy, encode_fn):
        self.encode_fn = encode_fn
        self.n_input = len(model.inputs)

        def reform(*row):
            # return (*row),
            if self.n_input == 4:
                x = row[0], row[1], row[2], row[3]
            elif self.n_input == 2:
                x = row[0], row[1],
            else:
                raise ValueError
            return x,

        self.inner_predictor = KerasPredictHelper(model, strategy, reform)

    def predict(self, input_list):
        return self.inner_predictor.predict(input_list)

    def predict_es(self, input_list: List[EncodedSegmentIF]):
        payload = [x.get_input() for x in input_list]
        l_decision_list, g_decision_list = self.predict(payload)
        real_input_len = len(input_list)
        l_decision_list = l_decision_list[:real_input_len]
        return l_decision_list


def get_encode_fn(model, do_batch_shaping=True):
    tokenizer = get_tokenizer()
    for input_tensor in model.inputs:
        print(input_tensor, input_tensor.shape)
    max_seq_length1 = model.inputs[0].shape[1]
    max_seq_length2 = model.inputs[2].shape[1]

    def encode_fn(p_tokens, h_tokens):
        input_ids1, input_mask1, segment_ids1 = encode_single(tokenizer, p_tokens, max_seq_length1)
        input_ids2, input_mask2, segment_ids2 = encode_single(tokenizer, h_tokens, max_seq_length2)
        x = input_ids1, segment_ids1, input_ids2, segment_ids2
        if do_batch_shaping:
            x = tuple(map(batch_shaping, x))
        return x

    return encode_fn


def load_local_decision_model(model_path):
    model = load_model_by_dir_or_abs(model_path)
    local_decision_layer = get_local_decision_layer_from_model_by_shape(model)
    new_outputs = [local_decision_layer.output, model.outputs]
    model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return model


def get_local_decision_nlims(run_config: RunConfig2):
    model_path = run_config.eval_config.model_save_path
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        c_log.debug("Loading model from {} ...".format(model_path))
        model = load_local_decision_model(model_path)
        encode_fn = get_encode_fn(model, False)
        c_log.debug("Done")
        nlits: LocalDecisionNLIMS = LocalDecisionNLIMS(model, strategy, encode_fn)
    return nlits


def get_weighted_nlims(run_config: RunConfig2):
    model_path = run_config.eval_config.model_save_path
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        c_log.debug("Loading model from {} ...".format(model_path))
        model = load_model_by_dir_or_abs(model_path)
        local_decision_layer = get_local_decision_layer_from_model_by_shape(model)
        weight_output_layer = get_weight_layer_from_model_by_shape(model)
        new_outputs = [local_decision_layer.output, weight_output_layer.output, model.outputs]
        model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
        encode_fn = get_encode_fn(model, False)
        c_log.debug("Done")

        def reform(*row):
            x = row[0], row[1], row[2], row[3]
            return x,

        KerasPredictHelper(model, strategy, reform)
        nlits: LocalDecisionNLIMS = LocalDecisionNLIMS(model, strategy, encode_fn)
    return nlits