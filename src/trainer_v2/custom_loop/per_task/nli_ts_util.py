import numpy as np
import tensorflow as tf
from keras import backend as K

from data_generator2.segmented_enc.seg_encoder_common import encode_two_segments
from tlm.data_gen.base import get_basic_input_feature_as_list


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


def load_local_decision_nli(model_path):
    model = tf.keras.models.load_model(model_path)
    local_decision_layer_idx = 12
    local_decision_layer = model.layers[local_decision_layer_idx]
    print("Local decision layer", local_decision_layer.name)
    new_outputs = [local_decision_layer.output, model.outputs]
    fun = K.function([model.input, ], new_outputs)  # evaluation function
    return fun


def load_local_decision_nli_model(model_path):
    model = tf.keras.models.load_model(model_path)
    local_decision_layer_idx = 12
    local_decision_layer = model.layers[local_decision_layer_idx]
    new_outputs = [local_decision_layer.output, model.outputs]
    model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return model

