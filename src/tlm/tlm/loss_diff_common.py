import tensorflow as tf

from models.transformer import bert_common_v2 as bc
from models.transformer.bert_common_v2 import get_shape_list


def loss_to_prob_pair(loss):
    y0 = tf.exp(-loss)
    y1 = 1 - y0
    return tf.stack([y0, y1], -1)


class IndependentLossModel:
    def __init__(self, bert_config):

        initializer = bc.create_initializer(bert_config.initializer_range)
        self.layer1 = bc.dense(bert_config.hidden_size,
                          initializer,
                          bc.get_activation(bert_config.hidden_act))

        self.logit_dense1 = bc.dense(2, initializer)
        self.logit_dense2 = bc.dense(2, initializer)

        self.graph_built = False

    def build_predictions(self, input_tensor):
        if self.graph_built:
            raise Exception()
        with tf.compat.v1.variable_scope("project"):
            hidden = self.layer1(input_tensor)
        with tf.compat.v1.variable_scope("cls1"):
            self.logits1 = self.logit_dense1(hidden)
        with tf.compat.v1.variable_scope("cls2"):
            self.logits2 = self.logit_dense2(hidden)
        self.prob1 = tf.nn.softmax(self.logits1)[:, :, 0]
        self.prob2 = tf.nn.softmax(self.logits2)[:, :, 0]
        self.graph_built = True

    def train_modeling(self, input_tensor,
                       masked_lm_positions, masked_lm_weights,
                       loss_base, loss_target):
        if self.graph_built:
            raise Exception()
        batch_size, _, hidden_dims = get_shape_list(input_tensor)
        input_tensor = bc.gather_indexes(input_tensor, masked_lm_positions)
        input_tensor = tf.reshape(input_tensor, [batch_size, -1, hidden_dims])
        with tf.compat.v1.variable_scope("project"):
            hidden = self.layer1(input_tensor)

        def cross_entropy(logits, loss_label):
            gold_prob = loss_to_prob_pair(loss_label)
            logits = tf.reshape(logits, gold_prob.shape)

            per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
                gold_prob,
                logits,
                axis=-1,
                name=None
            )
            per_example_loss = tf.cast(masked_lm_weights, tf.float32) * per_example_loss
            losses = tf.reduce_sum(per_example_loss, axis=1)
            loss = tf.reduce_mean(losses)
            return loss, per_example_loss
        with tf.compat.v1.variable_scope("cls1"):
            self.logits1 = self.logit_dense1(hidden)
        with tf.compat.v1.variable_scope("cls2"):
            self.logits2 = self.logit_dense2(hidden)



        self.loss1, self.per_example_loss1 = cross_entropy(self.logits1, loss_base)
        self.loss2, self.per_example_loss2 = cross_entropy(self.logits2, loss_target)

        self.prob1 = tf.nn.softmax(self.logits1)[:, :, 0]
        self.prob2 = tf.nn.softmax(self.logits2)[:, :, 0]

        self.total_loss = self.loss1 + self.loss2
        self.graph_built = True


def get_loss_independently(bert_config, input_tensor,
                           masked_lm_positions, masked_lm_weights, loss_base, loss_target):
    input_tensor = bc.gather_indexes(input_tensor, masked_lm_positions)

    hidden = bc.dense(bert_config.hidden_size,
                      bc.create_initializer(bert_config.initializer_range),
                      bc.get_activation(bert_config.hidden_act))(input_tensor)


    def get_regression_and_loss(hidden_vector, loss_label):
        logits = bc.dense(2, bc.create_initializer(bert_config.initializer_range))(hidden_vector)
        print("logits", logits.shape)
        gold_prob = loss_to_prob_pair(loss_label)
        print("gold_prob", gold_prob.shape)
        logits = tf.reshape(logits, gold_prob.shape)

        per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
            gold_prob,
            logits,
            axis=-1,
            name=None
        )
        per_example_loss = tf.cast(masked_lm_weights, tf.float32) * per_example_loss
        losses = tf.reduce_sum(per_example_loss, axis=1)
        loss = tf.reduce_mean(losses)

        return loss, per_example_loss, logits

    loss1, per_example_loss1, logits1 = get_regression_and_loss(hidden, loss_base)
    loss2, per_example_loss2, logits2 = get_regression_and_loss(hidden, loss_target)

    prob1 = tf.nn.softmax(logits1)[:, :, 0]
    prob2 = tf.nn.softmax(logits2)[:, :, 0]

    total_loss = loss1 + loss2
    return total_loss, loss1, loss2, per_example_loss1, per_example_loss2, prob1, prob2


def get_diff_loss(bert_config, input_tensor,
                           masked_lm_positions, masked_lm_weights, loss_base, loss_target):
    base_prob = tf.exp(-loss_base)
    target_prob = tf.exp(-loss_target)

    prob_diff = base_prob - target_prob

    input_tensor = bc.gather_indexes(input_tensor, masked_lm_positions)
    with tf.compat.v1.variable_scope("diff_loss"):

        hidden = bc.dense(bert_config.hidden_size,
                          bc.create_initializer(bert_config.initializer_range),
                          bc.get_activation(bert_config.hidden_act))(input_tensor)

        logits = bc.dense(1, bc.create_initializer(bert_config.initializer_range))(hidden)
        logits = tf.reshape(logits, prob_diff.shape)

    per_example_loss = tf.abs(prob_diff-logits)
    per_example_loss = tf.cast(masked_lm_weights, tf.float32) * per_example_loss
    losses = tf.reduce_sum(per_example_loss, axis=1)
    loss = tf.reduce_mean(losses)

    return loss, per_example_loss, logits


def recover_mask(input_ids, masked_lm_positions, masked_lm_ids):
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    flat_input_ids = tf.reshape(input_ids, [-1])
    offsets = tf.range(batch_size) * seq_length
    masked_lm_positions = masked_lm_positions + tf.expand_dims(offsets, 1)
    masked_lm_positions = tf.reshape(masked_lm_positions, [-1, 1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])

    output = tf.tensor_scatter_nd_update(flat_input_ids, masked_lm_positions, masked_lm_ids)
    output = tf.reshape(output, [batch_size, seq_length])
    output = tf.concat([input_ids[:, :1], output[:,1:]], axis=1)
    return output


def get_gold_diff(loss_base, loss_target):
    prob_base = tf.exp(-loss_base)
    prob_target = tf.exp(-loss_target)
    return prob_base - prob_target