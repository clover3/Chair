from typing import Any, NamedTuple

import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list2, create_attention_mask_from_input_mask
from tlm.model.bert_mask import BertModelMasked
from tlm.qtype.partial_relevance.attention_based.hard_concrete_optimize.bert_common_util import BertRunConfig, \
    InputStructureBase, BertCommonHP


def reduce_sum(t, axis_list):
    axis_list.sort(reverse=True)
    for axis in axis_list:
        t = tf.reduce_sum(t, axis=axis)
    return t


def get_valid_region_mask(mask):
    from_shape = get_shape_list2(mask)
    batch_size = from_shape[0]
    seq_length = from_shape[1]

    a = tf.cast(
        tf.reshape(mask, [batch_size, 1, seq_length]), tf.float32)
    b = tf.cast(
        tf.reshape(mask, [batch_size, seq_length, 1]), tf.float32)
    return a * b


Tensor = Any


class SampleRepeated(NamedTuple):
    m_input_ids: Tensor
    m_input_mask: Tensor
    m_segment_ids: Tensor
    m_attention_mask: Tensor


class SampleRepeatHelper:
    def __init__(self, batch_size, seq_length, n_sample):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_sample = n_sample

    def sample_repeat_from_input_structure(self, inputs: InputStructureBase, samples_attention_mask) -> SampleRepeated:
        base_attention_mask = create_attention_mask_from_input_mask(
            inputs.get_input_ids(), inputs.get_input_mask())
        return self.sample_repeat(inputs.get_input_ids(),
                                  inputs.get_input_mask(),
                                  inputs.get_segment_ids(),
                                  base_attention_mask,
                                  samples_attention_mask
                                  )

    def sample_repeat(self, input_ids, input_mask, segment_ids,
                      base_attention_mask, samples_attention_mask) -> SampleRepeated:
        base_attention_mask_ex = tf.expand_dims(base_attention_mask, 1)

        n_sample = self.n_sample
        batch_size = self.batch_size
        seq_length = self.seq_length

        attention_mask = tf.concat([base_attention_mask_ex, samples_attention_mask], axis=1)

        def repeat(input_ids_like):
            t = tf.expand_dims(input_ids_like, 1)
            t = tf.tile(t, [1, (n_sample + 1), 1])
            return t

        def flatten(t_3d):
            return tf.reshape(t_3d, [batch_size * (n_sample + 1), seq_length])

        def repeat_flatten(input_ids_like):
            t = repeat(input_ids_like)
            return flatten(t)

        m_input_ids = repeat_flatten(input_ids)
        m_input_mask = repeat_flatten(input_mask)
        m_segment_ids = repeat_flatten(segment_ids)
        m_attention_mask = tf.reshape(attention_mask, [batch_size * (n_sample + 1), seq_length, seq_length])
        return SampleRepeated(m_input_ids, m_input_mask, m_segment_ids, m_attention_mask)

    def split_logits(self, logits):
        logits_4d = tf.reshape(logits, [self.batch_size, self.n_sample+1, -1])
        base_logits = logits_4d[:, 0:1, :]
        masked_logits = logits_4d[:, 1:, :]
        return base_logits, masked_logits


def init_feed_bert_mask(bert_run_config: BertRunConfig, rpt, num_classes):
    model = BertModelMasked(
        config=bert_run_config.bert_config,
        is_training=bert_run_config.is_training,
        attention_mask=rpt.m_attention_mask,
        input_ids=rpt.m_input_ids,
        input_mask=rpt.m_input_mask,
        token_type_ids=rpt.m_segment_ids,
        use_one_hot_embeddings=bert_run_config.use_one_hot_embeddings)
    pooled_output = model.get_pooled_output()
    logits = tf.keras.layers.Dense(num_classes, name="cls_dense")(pooled_output)
    return model, logits


def get_fidelity_loss(base_logits, masked_logits):
    error_v = base_logits - masked_logits  # [batch_size, n_sample, num_classes]
    loss_1 = tf.norm(error_v, axis=-1)  # [batch_size, n_sample]
    return loss_1


def get_compact_loss(always_active_mask_ex, input_mask, samples_attention_mask):
    valid_mask = get_valid_region_mask(input_mask)
    valid_mask_ex = tf.expand_dims(valid_mask, 1)
    actually_used_mask = samples_attention_mask * valid_mask_ex
    non_default_selected_mask = actually_used_mask - always_active_mask_ex
    return reduce_sum(non_default_selected_mask, [2, 3])


def get_compact_loss_from_inputs(inputs, samples_attention_mask):
    return get_compact_loss(inputs.get_always_active_ex(), inputs.get_input_mask(), samples_attention_mask)




class AttnOptHP(BertCommonHP):
    n_samples = 8
    init_log_alpha = 0.0
    batch_size = 1  # alias = N
    lr = 1e-1  # learning rate. In paper, learning rate is adjusted to the global step.
    factor = 0.15
