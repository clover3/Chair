# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

import params_flow as pf
from bert.embeddings import BertEmbeddingsLayer
from bert.layer import Layer
from bert.loader import map_to_stock_variable_name, _checkpoint_exists
from tensorflow import keras

from trainer_v2.bert_for_tf2.w_mask.transformer import TransformerEncoderLayerWMask
from trainer_v2.custom_loop.modeling_common.bert_common import _load_stock_weights


class BertModelLayerWMask(Layer):
    """
    Implementation of BERT (arXiv:1810.04805), adapter-BERT (arXiv:1902.00751) and ALBERT (arXiv:1909.11942).

    See: https://arxiv.org/pdf/1810.04805.pdf - BERT
         https://arxiv.org/pdf/1902.00751.pdf - adapter-BERT
         https://arxiv.org/pdf/1909.11942.pdf - ALBERT

    """
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayerWMask.Params):
        pass

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )
        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayerWMask.from_params(
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
        super(BertModelLayerWMask, self).build(input_shape)

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

    def call(self, inputs, attention_mask, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        output           = self.encoders_layer(embedding_output, mask=mask,
                                               attention_mask=attention_mask,
                                               training=training)
        return output   # [B, seq_len, hidden_size]




def load_stock_weights_for_wmask(bert: BertModelLayerWMask, ckpt_path,
                       map_to_stock_fn=map_to_stock_variable_name,
                       n_expected_restore=None,
                       ):
    assert isinstance(bert, BertModelLayerWMask), "Expecting a BertModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(bert.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
                                  "Please add the layer in a Keras model and call model.build() first!"

    skipped_weight_value_tuples = _load_stock_weights(bert, ckpt_path, map_to_stock_fn, n_expected_restore)

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)



def load_stock_weights_nc(bert, ckpt_path,
                          map_to_stock_fn=map_to_stock_variable_name,
                          n_expected_restore=None,
                          ):
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(bert.weights) > 0, "Layer weights have not been instantiated yet. " \
                                  "Please add the layer in a Keras model and call model.build() first!"

    skipped_weight_value_tuples = _load_stock_weights(bert, ckpt_path, map_to_stock_fn, n_expected_restore)

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)
