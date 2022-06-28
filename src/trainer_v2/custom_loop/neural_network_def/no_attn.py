# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

import params_flow as pf
from bert.embeddings import BertEmbeddingsLayer
from bert.layer import Layer
from tensorflow import keras

from trainer_v2.bert_for_tf2.transformer import ProjectionLayer, TransformerSelfAttentionLayer


class SingleTransformerEncoderLayerWOAttn(Layer):
    """
    Multi-headed, single layer for the Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(TransformerSelfAttentionLayer.Params, ProjectionLayer.Params):
        intermediate_size       = None
        intermediate_activation = "gelu"

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads

        self.intermediate_layer   = None
        self.output_projector     = None

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)  # [B, seq_len, hidden_size]
        self.intermediate_layer = keras.layers.Dense(
            name="intermediate",
            units=self.params.intermediate_size,
            activation=self.get_activation(self.params.intermediate_activation),
            kernel_initializer=self.create_initializer()
        )
        self.output_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )

        super(SingleTransformerEncoderLayerWOAttn, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        attention_output    = layer_input

        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)

        # output
        layer_output = self.output_projector([intermediate_output, attention_output], mask=mask)

        return layer_output


class TransformerEncoderLayerWOAttn(Layer):
    """
    Multi-headed, multi-layer Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    Implemented for BERT, with support for ALBERT (sharing encoder layer params).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(SingleTransformerEncoderLayerWOAttn.Params):
        num_layers     = None
        out_layer_ndxs = None   # [-1]

        shared_layer   = False  # False for BERT, True for ALBERT

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.encoder_layers   = []
        self.shared_layer     = None  # for ALBERT
        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        # create all transformer encoder sub-layers
        if self.params.shared_layer:
            # ALBERT: share params
            self.shared_layer = SingleTransformerEncoderLayerWOAttn.from_params(self.params, name="layer_shared")
        else:
            # BERT
            for layer_ndx in range(self.params.num_layers):
                encoder_layer = SingleTransformerEncoderLayerWOAttn.from_params(
                    self.params,
                    name="layer_{}".format(layer_ndx),
                )
                self.encoder_layers.append(encoder_layer)

        super(TransformerEncoderLayerWOAttn, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_output = inputs

        layer_outputs = []
        for layer_ndx in range(self.params.num_layers):
            encoder_layer = self.encoder_layers[layer_ndx] if self.encoder_layers else self.shared_layer
            layer_input = layer_output
            layer_output = encoder_layer(layer_input, mask=mask, training=training)
            layer_outputs.append(layer_output)

        if self.params.out_layer_ndxs is None:
            # return the final layer only
            final_output = layer_output
        else:
            final_output = []
            for ndx in self.params.out_layer_ndxs:
                final_output.append(layer_outputs[ndx])
            final_output = tuple(final_output)

        return final_output



class BertModelLayerNoAttn(Layer):
    """
    Implementation of BERT (arXiv:1810.04805), adapter-BERT (arXiv:1902.00751) and ALBERT (arXiv:1909.11942).

    See: https://arxiv.org/pdf/1810.04805.pdf - BERT
         https://arxiv.org/pdf/1902.00751.pdf - adapter-BERT
         https://arxiv.org/pdf/1909.11942.pdf - ALBERT

    """
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayerWOAttn.Params):
        pass

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )
        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayerWOAttn.from_params(
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
        super(BertModelLayerNoAttn, self).build(input_shape)

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

    def call(self, inputs, mask=None, training=None):
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        output           = self.encoders_layer(embedding_output, mask=mask,
                                               training=training)
        return output   # [B, seq_len, hidden_size]

