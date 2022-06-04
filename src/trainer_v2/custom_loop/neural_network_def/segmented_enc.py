import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input


def split_stack_flatten_encode_stack(encoder, input_list,
                                     total_seq_length, window_length):
    num_window = int(total_seq_length / window_length)
    assert total_seq_length % window_length == 0
    batch_size, _ = get_shape_list2(input_list[0])

    def r3to2(arr):
        return tf.reshape(arr, [batch_size * num_window, window_length])

    input_list_stacked = split_stack_input(input_list, total_seq_length, window_length)
    input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
    rep_flatten = encoder.call(input_list_flatten)  # [batch_size * num_window, dim]
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


#  Add bias
class FuzzyLogicLayer2(tf.keras.layers.Layer):
    def __init__(self):
        super(FuzzyLogicLayer2, self).__init__()
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(
            initial_value=b_init(shape=(3,), dtype="float32"), trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        local_decisions = inputs
        local_entail_p = local_decisions[:, :, 0]
        local_neutral_p = local_decisions[:, :, 1]
        local_contradiction_p = local_decisions[:, :, 2]

        comb_contradiction_s = tf.reduce_max(local_contradiction_p, axis=-1)

        cnp1 = tf.reduce_max(local_neutral_p, axis=-1)  # [batch_size]
        cnp2 = 1 - comb_contradiction_s
        comb_neutral_s = tf.multiply(cnp1, cnp2)
        comb_entail_s = tf.math.exp(tf.reduce_mean(tf.math.log(local_entail_p), axis=-1))

        score_stacked = tf.stack([comb_entail_s, comb_neutral_s, comb_contradiction_s], axis=1)
        score_stacked = score_stacked + self.bias

        sum_s = tf.reduce_sum(score_stacked, axis=1, keepdims=True)
        sentence_logits = tf.divide(score_stacked, sum_s)
        sentence_prob = tf.nn.softmax(sentence_logits, axis=1)
        return sentence_prob

#
# class BERTAsymmetricContextualizedSlice:
#     def __init__(self, bert_params, config: ModelConfig2SegProject, combine_local_decisions_layer):
#         encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
#         encoder2_b = MeanProjectionEnc(bert_params, config.project_dim, "encoder2_b")
#         encoder2_a = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
#         max_seq_len1 = config.max_seq_length1
#         max_seq_len2 = config.max_seq_length2
#         num_classes = config.num_classes
#
#         l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
#         l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
#
#         l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
#         l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
#
#         rep1 = encoder1([l_input_ids1, l_token_type_ids1])
#         rep2_middle = encoder2_a([l_input_ids2, l_token_type_ids2], training=False)
#
#         rep_middle0, rep_middle1 = SplitSegmentIDLayer()(rep2_middle, l_input_ids2, l_token_type_ids2)
#         rep_middle_concat = tf.concat([rep_middle0, rep_middle1], axis=0)
#         rep2 = encoder2_b(rep_middle_concat)
#
#         vtf = VectorThreeFeature()
#         feature_rep = vtf(rep1, rep2)
#
#         hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
#         local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
#         self.local_decisions = local_decisions
#         combine_local_decisions = combine_local_decisions_layer()
#         output = combine_local_decisions(local_decisions)
#
#         inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
#         model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
#         self.model: keras.Model = model
#         self.l_bert_list = [encoder1.l_bert, encoder2_a, encoder2_b.l_bert]
#

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