
# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

from typing import NamedTuple

import params_flow as pf
import tensorflow as tf
from bert.embeddings import BertEmbeddingsLayer
from bert.layer import Layer
from tensorflow import keras

from trainer_v2.bert_for_tf2.spit_attn_probs.transformer import TransformerEncoderLayerSAP


class BertModelLayerSAP(Layer):
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayerSAP.Params):
        pass

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )
        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayerSAP.from_params(
            self.params,
            name="encoder"
        )

        self.support_masking  = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)
        super(BertModelLayerSAP, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, _ = input_shape
        else:
            input_ids_shape = input_shape

        output_shape = list(input_ids_shape) + [self.params.hidden_size]
        return output_shape

    def apply_adapter_freeze(self):
        """ Should be called once the model has been built to freeze
        all bet the adapter and layer normalization layers in BERT.
        """
        if self.params.adapter_size is not None:
            def freeze_selector(layer):
                return layer.name not in ["adapter-up", "adapter-down", "LayerNorm", "extra_word_embeddings"]
            pf.utils.freeze_leaf_layers(self, freeze_selector)

    def call(self, inputs, attention_mask=None, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        output, attention_probs_list           = self.encoders_layer(embedding_output, mask=mask,
                                               attention_mask=attention_mask,
                                               training=training)
        return output, attention_probs_list   # [B, seq_len, hidden_size]


class BertClsSAP(NamedTuple):
    l_bert: tf.keras.layers.Layer
    pooler: tf.keras.layers.Dense

    def apply(self, inputs):
        seq_out, attn_probs = self.l_bert(inputs)
        cls = self.pooler(seq_out[:, 0, :])
        return cls
