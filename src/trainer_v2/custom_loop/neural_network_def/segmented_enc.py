import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input, TwoLayerDense
from trainer_v2.custom_loop.train_loop_helper import eval_tensor


def split_stack_flatten_encode_stack(encoder, input_list,
                                     total_seq_length, window_length):
    num_window = int(total_seq_length / window_length)
    assert total_seq_length % window_length == 0
    batch_size, _ = get_shape_list2(input_list[0])

    def r3to2(arr):
        return tf.reshape(arr, [batch_size * num_window, window_length])

    input_list_stacked = split_stack_input(input_list, total_seq_length, window_length)
    input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
    rep_flatten = encoder(input_list_flatten)  # [batch_size * num_window, dim]
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


# Input [batch_size, num_window, num_classes]
# Output [batch_size, num_classes]
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


class FuzzyLogicLayerOnLogits(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        local_decisions = inputs
        local_entail_s = local_decisions[:, :, 0]
        local_neutral_s = local_decisions[:, :, 1]
        local_contradiction_s = local_decisions[:, :, 2]

        combined_contradiction_s = tf.reduce_max(local_contradiction_s, axis=-1)

        cnp1 = tf.reduce_max(local_neutral_s, axis=-1)  # [batch_size]
        cnp2 = 1 - tf.nn.sigmoid(combined_contradiction_s)
        combined_neutral_s = tf.multiply(cnp1, cnp2)
        combined_entail_s = tf.reduce_mean(local_entail_s, axis=-1)

        score_stacked = tf.stack([combined_entail_s, combined_neutral_s, combined_contradiction_s], axis=1)
        sum_s = tf.reduce_sum(score_stacked, axis=1, keepdims=True)
        sentence_logits = tf.divide(score_stacked, sum_s)
        sentence_prob = tf.nn.softmax(sentence_logits, axis=1)
        return sentence_prob


class MLPLabelCombine(tf.keras.layers.Layer):
    def __init__(self, num_window, num_classes):
        super(MLPLabelCombine, self).__init__()
        self.input_size = num_window * num_classes
        hidden1 = num_window * num_classes * 100
        self.hidden1 = hidden1
        self.tld = TwoLayerDense(hidden1, num_classes)

    def call(self, inputs, *args, **kwargs):
        h = tf.reshape(inputs, [-1, self.input_size])
        return self.tld(h)


#  Add bias
class FuzzyLogicLayerBias(tf.keras.layers.Layer):
    def __init__(self):
        super(FuzzyLogicLayerBias, self).__init__()
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(
            initial_value=b_init(shape=(3,), dtype="float32"), trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        return combine_local_decision_by_fuzzy_logic(inputs)


#  Add bias
class FuzzyLogicToMLP(tf.keras.layers.Layer):
    def __init__(self, num_window, num_classes, train_step):
        super(FuzzyLogicToMLP, self).__init__()
        self.alpha = tf.Variable(initial_value=1.0, trainable=False, dtype=tf.float32)
        self.fll = FuzzyLogicLayer()
        self.mlp = MLPLabelCombine(num_window, num_classes)
        self.decay_steps = train_step

    def callback(self, arg):
        train_step_var = arg['step']
        step = tf.minimum(train_step_var, self.decay_steps)
        alpha = 1 - step / self.decay_steps
        alpha = tf.cast(alpha, tf.float32)
        c_log.debug("FuzzyLogicToMLP::callback alpha was %s", eval_tensor(self.alpha))
        self.alpha.assign(alpha)
        c_log.debug("FuzzyLogicToMLP::callback alpha is now %s", eval_tensor(self.alpha))

    def call(self, inputs, *args, **kwargs):
        out1 = self.fll(inputs)
        out2 = self.mlp(inputs)
        output = self.alpha * out1 + (-self.alpha + 1) * out2
        return output


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
