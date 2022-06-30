# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

import params_flow as pf
import tensorflow as tf
from bert.embeddings import BertEmbeddingsLayer, PositionEmbeddingLayer, EmbeddingsProjector
from bert.layer import Layer
from tensorflow import keras
from tensorflow.keras import backend as K

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





class BertEmbeddingsLayer(Layer):
    class Params(PositionEmbeddingLayer.Params,
                 EmbeddingsProjector.Params):
        vocab_size               = None
        use_token_type           = True
        use_position_embeddings  = True
        token_type_vocab_size    = 2
        hidden_size              = 768
        hidden_dropout           = 0.1

        extra_tokens_vocab_size  = None  # size of the extra (task specific) token vocabulary (using negative token ids)

        #
        # ALBERT support - set embedding_size (or None for BERT)
        #
        embedding_size               = None   # None for BERT, not None for ALBERT
        project_embeddings_with_bias = True   # in ALBERT - True for Google, False for brightmart/albert_zh
        project_position_embeddings  = True   # in ALEBRT - True for Google, False for brightmart/albert_zh

        mask_zero                    = False

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.word_embeddings_layer       = None
        self.extra_word_embeddings_layer = None   # for task specific tokens (negative token ids)
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.word_embeddings_projector_layer = None   # for ALBERT
        self.layer_norm_layer = None
        self.dropout_layer    = None

        self.support_masking = self.params.mask_zero

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

        # use either hidden_size for BERT or embedding_size for ALBERT
        embedding_size = self.params.hidden_size if self.params.embedding_size is None else self.params.embedding_size

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings"
            )

        # ALBERT word embeddings projection
        if self.params.embedding_size is not None:
            self.word_embeddings_projector_layer = EmbeddingsProjector.from_params(
                self.params, name="word_embeddings_projector")

        position_embedding_size = embedding_size if self.params.project_position_embeddings else self.params.hidden_size

        if self.params.use_token_type:
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.params.token_type_vocab_size,
                output_dim=position_embedding_size,
                mask_zero=False,
                name="token_type_embeddings"
            )
        if self.params.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbeddingLayer.from_params(
                self.params,
                name="position_embeddings",
                hidden_size=position_embedding_size
            )

        self.layer_norm_layer = pf.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.params.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        input_ids = tf.cast(input_ids, dtype=tf.int32)

        if self.extra_word_embeddings_layer is not None:
            token_mask   = tf.cast(tf.greater_equal(input_ids, 0), tf.int32)
            extra_mask   = tf.cast(tf.less(input_ids, 0), tf.int32)
            token_ids    = token_mask * input_ids
            extra_tokens = extra_mask * (-input_ids)
            token_output = self.word_embeddings_layer(token_ids)
            extra_output = self.extra_word_embeddings_layer(extra_tokens)
            embedding_output = tf.add(token_output,
                                      extra_output * tf.expand_dims(tf.cast(extra_mask, K.floatx()), axis=-1))
        else:
            embedding_output = self.word_embeddings_layer(input_ids)

        # ALBERT: for brightmart/albert_zh weights - project only token embeddings
        if not self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        if token_type_ids is not None:
            token_type_ids    = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            seq_len  = input_ids.shape.as_list()[1]
            emb_size = embedding_output.shape[-1]

            pos_embeddings = self.position_embeddings_layer(seq_len)
            # broadcast over all dimension except the last two [..., seq_len, width]
            broadcast_shape = [1] * (embedding_output.shape.ndims - 2) + [seq_len, emb_size]
            embedding_output += tf.reshape(pos_embeddings, broadcast_shape)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        # ALBERT: for google-research/albert weights - project all embeddings
        if self.params.project_position_embeddings:
            if self.word_embeddings_projector_layer:
                embedding_output = self.word_embeddings_projector_layer(embedding_output)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None

        if not self.support_masking:
            return None

        return tf.not_equal(input_ids, 0)
