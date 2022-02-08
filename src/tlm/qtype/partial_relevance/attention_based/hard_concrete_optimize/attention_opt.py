import numpy as np
import tensorflow as tf

from models.transformer.bert_common_v2 import create_attention_mask_from_input_mask
from tf_v2_support import placeholder
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.attnetion_opt_utils import \
    SampleRepeatHelper, init_feed_bert_mask, get_fidelity_loss, get_compact_loss_from_inputs
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.bert_common_util import InputStructureBase, \
    get_bert_run_config
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.hard_concrete import hard_concrete, \
    hard_concrete_inf, get_sample_accuracy


def build_limited_attention(base_attention_mask,
                            always_active_mask_ex,
                            predicted_mask):
    base_attention_mask_ex = tf.expand_dims(base_attention_mask, 1)

    # if always_active_mask_ex is 1, then 1
    selected_mask = tf.minimum(always_active_mask_ex + predicted_mask, 1)
    # if base_attention_mask_ex is 1, then selected_mask
    # if base_attention_mask_ex is 0, then 0
    attention_mask = base_attention_mask_ex * selected_mask
    return attention_mask


class InputStructureAttnOpt(InputStructureBase):
    def __init__(self, seq_length, is_inference=False):
        super(InputStructureAttnOpt, self).__init__(seq_length)
        self.always_active_mask = placeholder(tf.int64, [None, seq_length, seq_length])  # (batch_size, n_seq, n_seq)
        self.is_inference = is_inference
        if is_inference:
            self.given_mask = placeholder(tf.float32, [None, seq_length, seq_length])  # (batch_size, n_seq, n_seq)

    def get_always_active_ex(self):
        return tf.cast(tf.expand_dims(self.always_active_mask, 1), tf.float32)

    def get_x_dict(self):
        d = super(InputStructureAttnOpt, self).get_x_dict()
        d['always_active_mask'] = self.always_active_mask
        if self.is_inference:
            d["given_mask"] = self.given_mask
        return d

    def get_given_mask_ex(self):
        return tf.expand_dims(self.given_mask, 1)


def get_valid_region_mask_np(mask):
    from_shape = mask.shape
    batch_size = from_shape[0]
    seq_length = from_shape[1]

    a = np.reshape(mask, [batch_size, 1, seq_length])
    b = np.reshape(mask, [batch_size, seq_length, 1])
    return a * b


class TransformerAttentionOptModel:
    def __init__(self, hp):
        num_classes = hp.num_classes
        is_training = False
        n_samples = hp.n_samples
        bert_run_config = get_bert_run_config(hp, is_training)
        seq_length = hp.seq_max
        batch_size = hp.batch_size
        inputs = InputStructureAttnOpt(seq_length)
        base_attention_mask = create_attention_mask_from_input_mask(
            inputs.input_ids, inputs.input_mask)

        sample_repeat_helper = SampleRepeatHelper(batch_size, seq_length, n_samples)
        self.x_dict = inputs.get_x_dict()
        init_val = np.ones([batch_size, seq_length, seq_length]) * hp.init_log_alpha
        log_alpha = tf.Variable(init_val, name="log_alpha", dtype=tf.float32)  # (batch_size, n_seq, n_seq)

        predicted_mask = hard_concrete(log_alpha, n_samples)  # (batch_size, n_sample, n_seq, n_seq)
        inf_predicted_mask = hard_concrete_inf(log_alpha)

        samples_attention_mask = build_limited_attention(base_attention_mask,
                                                         inputs.get_always_active_ex(),
                                                         predicted_mask)
        inf_attention_mask = build_limited_attention(base_attention_mask,
                                                         inputs.get_always_active_ex(),
                                                     tf.expand_dims(inf_predicted_mask, 1))

        rpt = sample_repeat_helper.sample_repeat_from_input_structure(inputs, samples_attention_mask)
        model, logits = init_feed_bert_mask(bert_run_config, rpt, num_classes)
        base_logits, masked_logits = sample_repeat_helper.split_logits(logits)
        sample_accuracy = get_sample_accuracy(base_logits, masked_logits)
        loss_1 = get_fidelity_loss(base_logits, masked_logits)
        loss_2 = get_compact_loss_from_inputs(inputs, samples_attention_mask)
        factor = sample_accuracy * hp.factor
        loss_arr = loss_1 + loss_2 * factor
        self.log_alpha = log_alpha
        self.loss = tf.reduce_mean(loss_arr)
        self.loss_1 = tf.reduce_mean(loss_1)
        self.loss_2 = tf.reduce_mean(loss_2)
        self.sample_accuracy = tf.reduce_mean(sample_accuracy)
        self.logits = logits
        self.inf_predicted_mask = inf_predicted_mask
        self.predicted_mask = predicted_mask

        self.base_logits = base_logits
        self.masked_logits = masked_logits
        self.inf_attention_mask = inf_attention_mask
        self.base_attention_mask = base_attention_mask
        self.samples_attention_mask = samples_attention_mask

    def get_debug_vars(self):
        return {
            'predicted_mask': self.predicted_mask,
            'samples_attention_mask': self.samples_attention_mask,
            'base_attention_mask': self.base_attention_mask,
            'inf_predicted_mask': self.inf_predicted_mask,
            'inf_attention_mask': self.inf_attention_mask,
            'always_active_mask': self.x_dict['always_active_mask'],
            'base_logits': self.base_logits,
            'masked_logits': self.masked_logits,
            'input_mask': self.x_dict['input_mask']
        }

    def debug_print(self, d):
        inf_attention_mask = d['inf_attention_mask']
        always_active_mask = d['always_active_mask']
        always_active_mask_ex = np.expand_dims(always_active_mask, 1)
        samples_attention_mask = d['samples_attention_mask']
        input_mask = d['input_mask']
        valid_mask = get_valid_region_mask_np(input_mask)
        valid_mask_ex = np.expand_dims(valid_mask, 1)
        actually_used_mask = samples_attention_mask * valid_mask_ex
        non_default_selected_mask = actually_used_mask - always_active_mask_ex

        def count_nonzero(m):
            return np.count_nonzero(np.less(1e-1, m))

        # print("Number of attention mask used")
        n_used_per_samples = [count_nonzero(non_default_selected_mask[0, i]) for i in range(8)]
        avg_use = count_nonzero(non_default_selected_mask) / 8
        # print("Samples " + ", ".join(map(str, n_used_per_samples)))
        n_inf_attention_mask = count_nonzero(inf_attention_mask)
        n_always_active_mask = count_nonzero(always_active_mask)
        n_used = n_inf_attention_mask - n_always_active_mask
        # print('Inference uses', n_used)
        # print(d['base_logits'])
        # print(d['masked_logits'])


class TransformerAttentionInfModel:
    def __init__(self, hp):
        num_classes = hp.num_classes
        bert_run_config = get_bert_run_config(hp, False)
        n_sample = 1

        seq_length = hp.seq_max
        batch_size = hp.batch_size
        inputs = InputStructureAttnOpt(seq_length, is_inference=True)
        base_attention_mask = create_attention_mask_from_input_mask(
            inputs.input_ids, inputs.input_mask)
        # Active mask indicate area that are
        srh = SampleRepeatHelper(batch_size, seq_length, n_sample)

        self.x_dict = inputs.get_x_dict()
        inference_attention_mask = build_limited_attention(base_attention_mask,
                                                         inputs.get_always_active_ex(),
                                                         inputs.get_given_mask_ex())
        rpt = srh.sample_repeat_from_input_structure(inputs, inference_attention_mask)
        model, logits = init_feed_bert_mask(bert_run_config, rpt, num_classes)

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
        self.base_logits = base_logits
        self.masked_logits = masked_logits


