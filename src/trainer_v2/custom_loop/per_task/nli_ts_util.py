from typing import List

import numpy as np
import tensorflow as tf
from keras import backend as K

from arg.qck.encode_common import encode_single
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import encode_two_segments, TwoSegConcatEncoder
from misc_lib import tprint, ceil_divide
from tlm.data_gen.base import get_basic_input_feature_as_list
from trainer_v2.custom_loop.demo.demo_common import EncodedSegmentIF
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200
from trainer_v2.custom_loop.runner.run_two_seg_concat import ModelConfig


def encode_prem(tokenizer, prem_text, max_seq_length1):
    tokens = tokenizer.tokenize(prem_text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer,
                                                                         max_seq_length1,
                                                                         tokens, segment_ids)
    return input_ids, segment_ids


def enum_hypo_tuples(tokenizer, hypo_text, window_size, segment_len):
    space_tokenized_tokens = hypo_text.split()
    st = 0

    def sb_tokenize(tokens):
        output = []
        for t in tokens:
            output.extend(tokenizer.tokenize(t))
        return output

    while st < len(space_tokenized_tokens):
        ed = st + window_size
        first_a = space_tokenized_tokens[:st]
        second = space_tokenized_tokens[st:ed]
        first_b = space_tokenized_tokens[ed:]

        first = sb_tokenize(first_a) + ["[MASK]"] + sb_tokenize(first_b)
        second = sb_tokenize(second)

        all_input_ids, all_input_mask, all_segment_ids = encode_two_segments(tokenizer, segment_len, first,
                                                                             second)
        yield all_input_ids, all_segment_ids
        st += window_size


def batch_shaping(item):
    arr = np.array(item)
    return np.expand_dims(arr, 0)


def load_local_decision_nli(model_path, get_local_decision_layer_from_model):
    model = tf.keras.models.load_model(model_path)
    model.summary()
    local_decision_layer = get_local_decision_layer_from_model(model)
    new_outputs = [local_decision_layer.output, model.outputs]
    fun = K.function([model.input, ], new_outputs)  # evaluation function
    return fun


def load_local_decision_nli_model(model_path, get_local_decision_layer_from_model):
    model = tf.keras.models.load_model(model_path)
    local_decision_layer = get_local_decision_layer_from_model(model)
    new_outputs = [local_decision_layer.output, model.outputs]
    model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return model


def get_local_decision_layer_from_model_by_shape(model):
    for idx, layer in enumerate(model.layers):
        shape = layer.output.shape
        try:
            if shape[1] == 2 and shape[2] == 3:
                print("Maybe this is local decision layer: {}".format(layer.name))
                return layer
        except IndexError:
            pass
    raise KeyError


class LocalDecisionNLICore:
    def __init__(self, model_path, strategy, n_input=4):
        tprint("Loading model...")
        self.predictor = load_local_decision_nli_model(model_path, get_local_decision_layer_from_model_by_shape)
        tprint("Done")
        self.strategy = strategy
        self.n_input = n_input

    def predict(self, input_list):
        batch_size = 16
        while len(input_list) % batch_size:
            input_list.append(input_list[-1])

        dataset = tf.data.Dataset.from_tensor_slices(input_list)
        strategy = self.strategy

        def reform(row):
            if self.n_input == 4:
                x = row[0], row[1], row[2], row[3]
            elif self.n_input == 2:
                x = row[0], row[1],
            else:
                raise ValueError
            return x,

        dataset = dataset.map(reform)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        maybe_step = ceil_divide(len(input_list), batch_size)
        dataset = distribute_dataset(strategy, dataset)
        model = self.predictor
        l_decision, g_decision = model.predict(dataset, steps=maybe_step)
        return l_decision

    def predict_es(self, input_list: List[EncodedSegmentIF]):
        payload = [x.get_input() for x in input_list]
        l_decision_list = self.predict(payload)
        real_input_len = len(input_list)
        l_decision_list = l_decision_list[:real_input_len]
        second_l_decision = [d[1] for d in l_decision_list]
        return second_l_decision


def get_two_seg_asym_encoder(do_batch_shaping=True):
    model_config = ModelConfig200_200()
    tokenizer = get_tokenizer()
    segment_len = int(model_config.max_seq_length2 / 2)

    def encode_two_seg_input(p_tokens, h_first, h_second):
        input_ids1, input_mask1, segment_ids1 = encode_single(tokenizer, p_tokens, model_config.max_seq_length1)
        triplet2 = encode_two_segments(tokenizer, segment_len, h_first, h_second)
        input_ids2, input_mask2, segment_ids2 = triplet2
        x = input_ids1, segment_ids1, input_ids2, segment_ids2
        if do_batch_shaping:
            x = tuple(map(batch_shaping, x))
        return x

    return encode_two_seg_input


def get_two_seg_concat_encoder():
    model_config = ModelConfig()
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, model_config.max_seq_length)

    begin = True

    def encode_two_seg_input(p_tokens, h_first, h_second):
        triplet = encoder.two_seg_concat_core(p_tokens, h_first, h_second)
        input_ids, input_mask, segment_ids = triplet
        x = input_ids, segment_ids
        nonlocal begin
        if begin:
            begin = False
        return x

    return encode_two_seg_input