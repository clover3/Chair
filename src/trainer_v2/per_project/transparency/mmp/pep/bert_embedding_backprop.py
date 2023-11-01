from trainer_v2.bert_for_tf2 import BertModelLayer

from __future__ import absolute_import, division, print_function

import params_flow as pf
from bert.embeddings import BertEmbeddingsLayer, EmbeddingsProjector, PositionEmbeddingLayer
from bert.layer import Layer
from tensorflow import keras

from trainer_v2.bert_for_tf2.transformer import TransformerEncoderLayer


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import params_flow as pf

from tensorflow import keras
from tensorflow.keras import backend as K

import bert




class CustomBertEmbeddingsLayer(BertEmbeddingsLayer):
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

        self.word_embeddings_layer = MyEmbeddingLayer(
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


class CustomBertModelLayer(BertModelLayer):
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embeddings_layer = CustomBertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )
        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayer.from_params(
            self.params,
            name="encoder"
        )

        self.support_masking  = True
