import numpy as np
import tensorflow as tf

from models.transformer.bert_common_v2 import create_attention_mask_from_input_mask, get_shape_list
from tf_v2_support import placeholder
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attnetion_opt_utils import SampleRepeatHelper, \
    init_feed_bert_mask, get_fidelity_loss, reduce_sum, get_valid_region_mask, \
    SampleRepeated, get_compact_loss_from_inputs
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.bert_common_util import InputStructureBase, \
    get_bert_run_config
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.hard_concrete import hard_concrete_inf, \
    get_sample_accuracy


def hard_concrete_1d(log_alpha, n_sample):
    eps = 1e-20
    batch_size, n_seq1, = get_shape_list(log_alpha)
    shape = [batch_size, n_sample, n_seq1] # (batch_size, n_sample, n_seq)
    u = tf.random.uniform(shape, minval=0, maxval=1)
    l = tf.math.log(u + eps) - tf.math.log(1 - u + eps)
    temperature = 0.2
    log_alpha_ex = tf.expand_dims(log_alpha, 1)
    y = log_alpha_ex + l
    gamma = -0.1
    zeta = 1.1
    z_hat = tf.nn.sigmoid(y / temperature)
    z_hat_shift = z_hat * (zeta - gamma) + gamma
    z = tf.minimum(tf.maximum(z_hat_shift, 0), 1)
    return z


def build_attention_mask_to_feed(linear_valid_target_mask_ex,
                                 base_attention_mask,
                                 always_active_mask_ex,
                                 seq_length,
                                 predicted_mask):
    predicted_mask_2 = linear_valid_target_mask_ex * predicted_mask
    m1 = tf.tile(tf.expand_dims(predicted_mask_2, 2), [1, 1, seq_length, 1])
    m2 = tf.tile(tf.expand_dims(predicted_mask_2, 3), [1, 1, 1, seq_length])
    m = m1 + m2
    selected_mask = tf.minimum(always_active_mask_ex + m, 1)
    base_attention_mask_ex = tf.expand_dims(base_attention_mask, 1)
    attention_mask = tf.minimum(base_attention_mask_ex, selected_mask)
    return attention_mask


class InputStructureQT(InputStructureBase):
    def __init__(self, seq_length, is_inference=False):
        super(InputStructureQT, self).__init__(seq_length)
        self.linear_valid_target_mask = placeholder(tf.int64, [None, seq_length])
        # linear_valid_target_mask: 1 when this is document (segment to search evidences from)
        #                           0 when this is query
        #

        self.always_active_mask = placeholder(tf.int64, [None, seq_length, seq_length])  # (batch_size, n_seq, n_seq)
        self.is_inference = is_inference
        if is_inference:
            self.given_mask = placeholder(tf.float32, [None, seq_length])  # (batch_size, n_seq, n_seq)

    def get_always_active_ex(self):
        return tf.cast(tf.expand_dims(self.always_active_mask, 1), tf.float32)

    def get_linear_valid_target_mask_ex(self):
        return tf.cast(tf.expand_dims(self.linear_valid_target_mask, 1), tf.float32)

    def get_x_dict(self):
        d = super(InputStructureQT, self).get_x_dict()
        d["always_active_mask"] = self.always_active_mask
        d["linear_valid_target_mask"] = self.linear_valid_target_mask
        if self.is_inference:
            d["given_mask"] = self.given_mask
        return d

    def get_given_mask_ex(self):
        return tf.expand_dims(self.given_mask, 1)


# Difference with TransformerAttentionOptModel
#   linear_valid_target_mask = placeholder(tf.int64, [None, seq_length])

def get_compact_loss_qt(always_active_mask_ex,
                        input_mask,
                        samples_attention_mask):

    valid_mask = get_valid_region_mask(input_mask)
    valid_mask_ex = tf.expand_dims(valid_mask, 1)
    non_default_selected_mask = samples_attention_mask * valid_mask_ex - always_active_mask_ex
    return reduce_sum(non_default_selected_mask, [2, 3])


class TransformerAttentionOptQTModel:
    def __init__(self, hp):
        num_classes = hp.num_classes
        is_training = False
        n_samples = hp.n_samples
        bert_run_config = get_bert_run_config(hp, is_training)
        seq_length = hp.seq_max
        batch_size = hp.batch_size
        inputs = InputStructureQT(seq_length)

        base_attention_mask = create_attention_mask_from_input_mask(
            inputs.input_ids, inputs.input_mask)
        sample_repeat_helper = SampleRepeatHelper(batch_size, seq_length, n_samples)

        self.x_dict = inputs.get_x_dict()
        init_val = np.ones([batch_size, seq_length]) * hp.init_log_alpha
        log_alpha = tf.Variable(init_val, name="log_alpha", dtype=tf.float32)  # (batch_size, n_seq)
        predicted_mask = hard_concrete_1d(log_alpha, n_samples)  # (batch_size, n_sample, n_seq)
        inf_predicted_mask = hard_concrete_inf(log_alpha)

        samples_attention_mask = build_attention_mask_to_feed(inputs.get_linear_valid_target_mask_ex(),
                                                              base_attention_mask,
                                                              inputs.get_always_active_ex(),
                                                              seq_length,
                                                              predicted_mask)

        repeated_inputs: SampleRepeated = sample_repeat_helper.sample_repeat_from_input_structure(inputs,
                                                 samples_attention_mask)
        model, logits = init_feed_bert_mask(bert_run_config, repeated_inputs, num_classes)
        base_logits, masked_logits = sample_repeat_helper.split_logits(logits)

        sample_accuracy = get_sample_accuracy(base_logits, masked_logits)
        loss_1 = get_fidelity_loss(base_logits, masked_logits)
        loss_2 = get_compact_loss_from_inputs(inputs, samples_attention_mask)

        factor = sample_accuracy * 0.05
        loss_arr = loss_1 + loss_2 * factor
        self.log_alpha = log_alpha
        self.loss = tf.reduce_mean(loss_arr)
        self.loss_1 = tf.reduce_mean(loss_1)
        self.loss_2 = tf.reduce_mean(loss_2)
        self.sample_accuracy = tf.reduce_mean(sample_accuracy)
        self.logits = logits
        self.inf_predicted_mask = inf_predicted_mask
        self.predicted_mask = predicted_mask
        self.base_attention_mask = base_attention_mask
        self.samples_attention_mask = samples_attention_mask

    def get_debug_vars(self):
        return {
            'predicted_mask': self.predicted_mask,
            'samples_attention_mask': self.samples_attention_mask,
            'base_attention_mask': self.base_attention_mask,
            'always_active_mask': self.x_dict['always_active_mask'],
        }

    def debug_print(self, d):
        samples_attention_mask = d['samples_attention_mask']
        always_active_mask = d['always_active_mask']
        base_attention_mask = d['base_attention_mask']

        def count_nonzero(m):
            return np.count_nonzero(np.less(1e-8, m))

        print("always_active_mask has {} items".format(count_nonzero(always_active_mask)))
        print("samples_attention_mask has {} items".format(count_nonzero(samples_attention_mask)))
        print("samples_attention_mask has {} items per sample".format(count_nonzero(samples_attention_mask) / 8))
        print("base_attention_mask has {} items".format(count_nonzero(base_attention_mask)))
        verify_samples_attention_mask(samples_attention_mask)


def near(v1, v2):
    return abs(v1- v2 ) < 1e-6


def print_samples_attention_mask(samples_attention_mask):
    print(samples_attention_mask.shape)
    n_samples = samples_attention_mask.shape[1] # [1, n_sample, seq_length, seq_length
    seq_length = samples_attention_mask.shape[2]
    seq_length = 120
    for i in range(n_samples):
        print("Sample ", i)
        for row in range(seq_length):
            consecutive_1 = 0
            row_summary = "row {}: ".format(row)
            for col in range(seq_length):
                v = samples_attention_mask[0, i, row, col]
                if near(v, 1.):
                    consecutive_1 += 1
                else:
                    if consecutive_1 > 4:
                        st = col - consecutive_1
                        row_summary += "All 1 [{}, {})".format(st, col)
                    consecutive_1 = 0

            n_nonzero = np.count_nonzero(samples_attention_mask[0, i, row])
            row_summary += ", {} non zero items".format(n_nonzero)
            print(row_summary)


def verify_samples_attention_mask(samples_attention_mask):
    print(samples_attention_mask.shape)
    n_samples = samples_attention_mask.shape[1] # [1, n_sample, seq_length, seq_length
    seq_length = samples_attention_mask.shape[2]
    seg1_l = 5
    seg2_l = 100
    seq_length = 120
    def get_token_type(i):
        if i == 0:
            return "CLS"
        elif i < seg1_l + 1:
            return "Query"
        elif i == seg1_l + 1:
            return "SEP"
        elif i < seg1_l + seg2_l + 2:
            return "Doc"
        elif i == seg1_l + seg2_l + 2:
            return "SEP"
        else:
            return "PAD"

    for i in range(n_samples):
        for row in range(seq_length):
            for col in range(seq_length):
                v = samples_attention_mask[0, i, row, col]
                if get_token_type(row) == "PAD":
                    pass
                elif get_token_type(col) == "PAD":
                    if not near(v, 0.0):
                        print(f"Cell {row}, {col} should have 0 but has {v}")
                elif get_token_type(row) in ["CLS", "SEP"] \
                        or get_token_type(col) in ["CLS", "SEP"] \
                        or get_token_type(row) == get_token_type(col) \
                        :
                    if not near(v, 1.0):
                        print(f"Cell {row}, {col} is {get_token_type(row)}, {get_token_type(col)}, it must be 1 but {v}")

                if v > 1 + 1e-5:
                    print(f"Cell {row}, {col} has {v}")


class TransformerAttentionQTInferenceModel:
    def __init__(self, hp, is_training=True):
        bert_run_config = get_bert_run_config(hp, is_training)
        seq_length = hp.seq_max
        batch_size = hp.batch_size
        num_classes = hp.num_classes

        inputs = InputStructureQT(seq_length, is_inference=True)
        base_attention_mask = create_attention_mask_from_input_mask(
            inputs.input_ids, inputs.input_mask)
        srh = SampleRepeatHelper(batch_size, seq_length, hp.n_sample)
        self.x_dict = inputs.get_x_dict()
        inference_attention_mask = build_attention_mask_to_feed(inputs.get_linear_valid_target_mask_ex(),
                                                              base_attention_mask,
                                                              inputs.get_always_active_ex(),
                                                              seq_length,
                                                              inputs.get_given_mask_ex())
        repeated_inputs: SampleRepeated = srh.sample_repeat_from_input_structure(inputs,
                                                                                 inference_attention_mask)
        model, logits = init_feed_bert_mask(bert_run_config, repeated_inputs, num_classes)
        base_logits, masked_logits = srh.split_logits(logits)
        sample_accuracy = get_sample_accuracy(base_logits, masked_logits)
        loss_1 = get_fidelity_loss(base_logits, masked_logits)
        loss_2 = get_compact_loss_from_inputs(inputs, inference_attention_mask)
        factor = sample_accuracy * 0.15
        loss_arr = loss_1 + loss_2 * factor
        self.loss = tf.reduce_mean(loss_arr)
        self.loss_1 = tf.reduce_mean(loss_1)
        self.loss_2 = tf.reduce_mean(loss_2)
        self.sample_accuracy = tf.reduce_mean(sample_accuracy)
        self.logits = logits
