# coding=utf-8
#
# created by kpe on 20.Mar.2019 at 16:30
#

from __future__ import absolute_import, division, print_function

from bert.layer import Layer
# from tensorflow.python import keras
from tensorflow import keras

from trainer_v2.bert_for_tf2.spit_attn_probs.attention import AttentionLayerSAP, AttentionLayerSAP_QK, \
    AttentionLayerSAP_V
from trainer_v2.bert_for_tf2.transformer import ProjectionLayer


class TransformerSelfAttentionLayerSAP(Layer):
    class Params(ProjectionLayer.Params,
                 AttentionLayerSAP.Params):
        hidden_size         = None
        num_heads           = None
        hidden_dropout      = None
        attention_dropout   = 0.1
        initializer_range   = 0.02

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads
        assert params.size_per_head is None or self.size_per_head == params.size_per_head

        self.attention_layer_a     = None
        self.attention_layer_b = None
        self.attention_projector = None

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        self.layer_a = AttentionLayerSAP_QK.from_params(self.params,
                                                        name="self",
                                                        size_per_head=self.size_per_head
                                                        )
        self.layer_b = AttentionLayerSAP_V.from_params(self.params,
                                                       name="self",
                                                       size_per_head=self.size_per_head
                                                       )
        self.attention_projector = ProjectionLayer.from_params(
            self.params,
            name="output",
        )

        super(TransformerSelfAttentionLayerSAP, self).build(input_shape)

    def call(self, inputs, attention_mask, mask=None, training=None):
        layer_input = inputs

        #
        # TODO: is it OK to recompute the 3D attention mask in each attention layer
        #
        attention_probs = self.layer_a(inputs, attention_mask)
        attention_head = self.layer_b([inputs, attention_probs])
        attention_output = self.attention_projector([attention_head, layer_input], mask=mask, training=training)
        return attention_output, attention_probs



class SingleTransformerEncoderLayerSAP(Layer):
    """
    Multi-headed, single layer for the Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(TransformerSelfAttentionLayerSAP.Params,
                 ProjectionLayer.Params):
        intermediate_size       = None
        intermediate_activation = "gelu"

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        params = self.params
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("The hidden_size:[{}] is not a multiple of num_heads:[{}]".format(params.hidden_size,
                                                                                               params.num_heads))
        self.size_per_head = params.hidden_size // params.num_heads

        self.self_attention_layer = None
        self.intermediate_layer   = None
        self.output_projector     = None

        self.supports_masking = True

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)  # [B, seq_len, hidden_size]

        self.self_attention_layer = TransformerSelfAttentionLayerSAP.from_params(
            self.params,
            name="attention"
        )
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

        super(SingleTransformerEncoderLayerSAP, self).build(input_shape)

    def call(self, inputs, attention_mask, mask=None, training=None):
        layer_input = inputs

        attention_output, attention_probs    = self.self_attention_layer(layer_input, attention_mask, mask=mask, training=training)

        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)

        # output
        layer_output = self.output_projector([intermediate_output, attention_output], mask=mask)

        return layer_output, attention_probs


class TransformerEncoderLayerSAP(Layer):
    """
    Multi-headed, multi-layer Transformer from 'Attention is All You Need' (arXiv: 1706.03762).

    Implemented for BERT, with support for ALBERT (sharing encoder layer params).

    See also: https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """

    class Params(SingleTransformerEncoderLayerSAP.Params):
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
            self.shared_layer = SingleTransformerEncoderLayerSAP.from_params(self.params, name="layer_shared")
        else:
            # BERT
            for layer_ndx in range(self.params.num_layers):
                encoder_layer = SingleTransformerEncoderLayerSAP.from_params(
                    self.params,
                    name="layer_{}".format(layer_ndx),
                )
                self.encoder_layers.append(encoder_layer)

        super(TransformerEncoderLayerSAP, self).build(input_shape)

    def call(self, inputs, attention_mask, mask=None, training=None):
        layer_output = inputs

        layer_outputs = []
        attention_probs_list = []
        for layer_ndx in range(self.params.num_layers):
            encoder_layer = self.encoder_layers[layer_ndx] if self.encoder_layers else self.shared_layer
            layer_input = layer_output

            layer_output, attention_probs = encoder_layer(layer_input, attention_mask, mask=mask, training=training)
            layer_outputs.append(layer_output)
            attention_probs_list.append(attention_probs)

        if self.params.out_layer_ndxs is None:
            # return the final layer only
            final_output = layer_output
        else:
            final_output = []
            for ndx in self.params.out_layer_ndxs:
                final_output.append(layer_outputs[ndx])
            final_output = tuple(final_output)

        return final_output, attention_probs_list


