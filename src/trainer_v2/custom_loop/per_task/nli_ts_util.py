import functools
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from arg.qck.encode_common import encode_single
from data_generator.tokenizer_wo_tf import get_tokenizer, sb_tokenize_w_tokenizer
from data_generator2.segmented_enc.seg_encoder_common import encode_two_segments, TwoSegConcatEncoder
from data_generator2.segmented_enc.two_seg_by_mask import ConcatWMasInference
from misc_lib import ceil_divide
from tlm.data_gen.base import get_basic_input_feature_as_list
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.per_task.ts_util import get_dataset_factory_two_seg, \
    get_local_decision_layer_from_model_by_shape
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs


class EncodedSegmentIF(ABC):
    @abstractmethod
    def get_input(self):
        pass


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


def load_local_decision_model_n_label_3(model_path, n_label=3):
    model = load_model_by_dir_or_abs(model_path)
    local_decision_layer = get_local_decision_layer_from_model_by_shape(model, n_label)
    new_outputs = [local_decision_layer.output, model.outputs]
    model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return model


class LocalDecisionNLICore:
    def __init__(self, model, strategy, encode_fn, batch_size=16):
        self.strategy = strategy
        self.model = model
        self.n_input = len(self.model.inputs)
        self.l_list: List[int] = [input_unit.shape[1] for input_unit in self.model.inputs]
        self.encode_fn = encode_fn
        self.batch_size = batch_size

        def get_spec(input_i: tf.keras.layers.Input):
            return tf.TensorSpec(input_i.type_spec.shape[1:], dtype=tf.int32)
        self.out_sig = [get_spec(input) for input in model.inputs]

    def predict(self, input_list: List[Tuple]):
        batch_size = self.batch_size

        while len(input_list) % batch_size:
            input_list.append(input_list[-1])

        def generator():
            yield from input_list

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tuple(self.out_sig))
        strategy = self.strategy

        def reform(*row):
            # return (*row),
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
        model = self.model
        verbose = 1 if maybe_step > 15 else 0
        l_decision, g_decision = model.predict(dataset, steps=maybe_step, verbose=verbose)
        return l_decision, g_decision

    def predict_es(self, input_list: List[EncodedSegmentIF]):
        payload = [x.get_input() for x in input_list]
        l_decision_list, g_decision_list = self.predict(payload)
        real_input_len = len(input_list)
        l_decision_list = l_decision_list[:real_input_len]
        second_l_decision = [d[1] for d in l_decision_list]
        return second_l_decision


class LocalDecisionNLICoreSecond:
    def __init__(self, model, strategy, encode_fn):
        self.strategy = strategy
        self.model = model
        self.n_input = len(self.model.inputs)
        self.l_list: List[int] = [input_unit.shape[1] for input_unit in self.model.inputs]
        self.encode_fn = encode_fn

        def get_spec(input_i: tf.keras.layers.Input):
            return tf.TensorSpec(input_i.type_spec.shape[1:], dtype=tf.int32)
        self.out_sig = [get_spec(input) for input in model.inputs]

    def predict(self, input_list):
        batch_size = 16
        while len(input_list) % batch_size:
            input_list.append(input_list[-1])

        def generator():
            yield from input_list

        # dataset = tf.data.Dataset.from_tensor_slices(input_list)
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tuple(self.out_sig))
        strategy = self.strategy

        def reform(*row):
            # return (*row),
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
        model = self.model
        l_decision = model.predict(dataset, steps=maybe_step)
        return l_decision

    def predict_es(self, input_list: List[EncodedSegmentIF]):
        payload = [x.get_input() for x in input_list]
        l_decision_list = self.predict(payload)
        print(l_decision_list.shape)
        print(l_decision_list)
        real_input_len = len(input_list)
        print(real_input_len)

        l_decision_list = l_decision_list[:real_input_len]
        return l_decision_list


def get_two_seg_asym_encoder(max_seq_length1, max_seq_length2, do_batch_shaping=True):
    tokenizer = get_tokenizer()
    segment_len = int(max_seq_length2 / 2)

    def encode_two_seg_input(p_tokens, h_first, h_second):
        input_ids1, input_mask1, segment_ids1 = encode_single(tokenizer, p_tokens, max_seq_length1)
        triplet2 = encode_two_segments(tokenizer, segment_len, h_first, h_second)
        input_ids2, input_mask2, segment_ids2 = triplet2
        x = input_ids1, segment_ids1, input_ids2, segment_ids2
        if do_batch_shaping:
            x = tuple(map(batch_shaping, x))
        return x

    return encode_two_seg_input


def get_two_seg_concat_encoder(max_seq_length):
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, max_seq_length)

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


def get_concat_mask_encoder(max_seq_length):
    tokenizer = get_tokenizer()
    encoder = ConcatWMasInference(tokenizer, max_seq_length)

    begin = True

    def encode_two_seg_input(p_tokens, h_tokens, mask):
        assert max(mask) <= 1
        triplet = encoder.encode(p_tokens, h_tokens, mask)
        input_ids, input_mask, segment_ids = triplet
        x = input_ids, segment_ids
        nonlocal begin
        if begin:
            begin = False
        return x

    return encode_two_seg_input


def enum_hypo_token_tuple_from_tokens(tokenizer, space_tokenized_tokens, window_size, offset=0) -> \
        List[Tuple[List[str], List[str], int, int]]:
    st = offset
    sb_tokenize = functools.partial(sb_tokenize_w_tokenizer, tokenizer)

    while st < len(space_tokenized_tokens):
        ed = st + window_size
        first_a = space_tokenized_tokens[:st]
        second = space_tokenized_tokens[st:ed]
        first_b = space_tokenized_tokens[ed:]

        first = sb_tokenize(first_a) + ["[MASK]"] + sb_tokenize(first_b)
        second = sb_tokenize(second)
        yield first, second, st, ed
        st += window_size


def enum_hypo_token_wmask(sb_tokenize, space_tokenized_tokens, window_size, offset=0) -> \
        List[Tuple[List[str], List[int], int, int]]:
    st = offset

    while st < len(space_tokenized_tokens):
        ed = st + window_size
        first_a = space_tokenized_tokens[:st]
        second = space_tokenized_tokens[st:ed]
        first_b = space_tokenized_tokens[ed:]

        first_a_sb = sb_tokenize(first_a)
        first_b_sb = sb_tokenize(first_b)
        second_sb = sb_tokenize(second)
        tokens = first_a_sb + second_sb + first_b_sb
        mask = [0] * len(first_a_sb) + [1] * len(second_sb) + [0] * len(first_b_sb)
        yield tokens, mask, st, ed
        st += window_size


def dataset_factory_600_3(payload: List):
    model_config = ModelConfig600_3()
    max_seq_length = model_config.max_seq_length
    dataset_factory = get_dataset_factory_two_seg(max_seq_length)
    return dataset_factory(payload)


