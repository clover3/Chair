import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input, VectorThreeFeature, \
    MeanProjectionEnc
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


def split_stack_flatten_encode_stack(encoder, input_list,
                                     total_seq_length, window_length):
    num_window = int(total_seq_length / window_length)
    assert total_seq_length % window_length == 0
    batch_size, _ = get_shape_list2(input_list[0])

    def r3to2(arr):
        return tf.reshape(arr, [batch_size * num_window, window_length])

    input_list_stacked = split_stack_input(input_list, total_seq_length, window_length)
    input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
    rep_flatten = encoder.apply(input_list_flatten)  # [batch_size * num_window, dim]
    _, rep_dim = get_shape_list2(rep_flatten)

    def r2to3(arr):
        return tf.reshape(arr, [batch_size, num_window, rep_dim])

    rep_stacked = r2to3(rep_flatten)
    return rep_stacked


class StackedInputMapper(tf.keras.layers.Layer):
    def __init__(self, encoder, total_seq_length, window_length):
        super(StackedInputMapper, self).__init__()
        self.encoder = encoder
        self.total_seq_length = total_seq_length
        self.window_length = window_length

    def call(self, inputs, *args, **kwargs):
        return split_stack_flatten_encode_stack(self.encoder, inputs,
                                                self.total_seq_length, self.window_length)




def combine_local_decision_by_fuzzy_logic(local_decisions):
    local_entail_p = local_decisions[:, :, 0]
    local_neutral_p = local_decisions[:, :, 1]
    local_contradiction_p = local_decisions[:, :, 2]

    combined_contradiction_s = tf.reduce_max(local_contradiction_p, axis=-1)

    cnp1 = tf.reduce_max(local_neutral_p, axis=-1)  # [batch_size]
    cnp2 = 1-combined_contradiction_s
    combined_neutral_s = tf.multiply(cnp1, cnp2)
    combined_entail_s = tf.math.exp(tf.reduce_mean(tf.math.log(local_entail_p), axis=-1))

    score_stacked = tf.stack([combined_entail_s, combined_neutral_s, combined_contradiction_s], axis=1)
    sum_s = tf.reduce_sum(score_stacked, axis=1, keepdims=True)
    sentence_logits = tf.divide(score_stacked, sum_s)
    sentence_prob = tf.nn.softmax(sentence_logits, axis=1)
    return sentence_prob


class FuzzyLogicLayer(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return combine_local_decision_by_fuzzy_logic(inputs)


class BERTEvenSegmented:
    def __init__(self, bert_params, config: ModelConfig2SegProject, combine_local_decisions_layer):
        num_window = 2
        # encoder1, encoder2 = get_two_projected_mean_encoder(bert_params, config.project_dim)
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        encoder2 = MeanProjectionEnc(bert_params, config.project_dim, "encoder2")
        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2

        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        # [batch_size, dim]
        rep1 = encoder1.apply([l_input_ids1, l_token_type_ids1])

        window_length = int(max_seq_len2 / num_window)
        inputs_for_seg2 = [l_input_ids2, l_token_type_ids2]

        mapper = StackedInputMapper(encoder2, max_seq_len2, window_length)
        # [batch_size, num_window, dim]
        # rep2_stacked = split_stack_flatten_encode_stack(encoder2, inputs_for_seg2, max_seq_len2, window_length)
        rep2_stacked = StackedInputMapper(encoder2, max_seq_len2, window_length)(inputs_for_seg2)

        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, num_window, 1])

        # [batch_size, num_window, dim2 ]
        feature_rep = VectorThreeFeature()((rep1_stacked, rep2_stacked))
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        combine_local_decisions = combine_local_decisions_layer()
        output = combine_local_decisions(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2.l_bert]


def main():
    sent1 = [
        [0.9, 0.0, 0.1],
        [0.1, 0.0, 0.9],
    ]
    sent2 = [
        [0.9, 0.0, 0.1],
        [0.9, 0.0, 0.1],
    ]
    batch = tf.constant([sent1, sent2])
    out_prob = combine_local_decision_by_fuzzy_logic(batch)
    for i in range(len(batch)):
        print(batch[i].numpy(), out_prob[i].numpy())


if __name__ == "__main__":
    main()