from __future__ import absolute_import, division, print_function

import params_flow as pf
import tensorflow as tf
from bert.embeddings import BertEmbeddingsLayer, EmbeddingsProjector, PositionEmbeddingLayer
from tensorflow import keras

from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.transformer import TransformerEncoderLayer


class MyExtraEmbedding(keras.layers.Embedding):
    def __init__(self, **kwargs):
        super(MyExtraEmbedding, self).__init__(**kwargs)

    def set_target_emb(self, target_emb):
        self.target_emb: tf.keras.layers.Layer = target_emb

    def call(self, input_ids):
        key_emb = super(MyExtraEmbedding, self).call(input_ids)  # [B, M, H]
        mask = tf.expand_dims(tf.cast(tf.not_equal(0, input_ids), tf.float32) , axis=2)
        key_emb = key_emb * mask

        emb_vector = self.target_emb.weights[0]  # [V, H]
        zero_vector = emb_vector[0, :]
        weights = tf.matmul(key_emb, emb_vector, transpose_b=True)  # [B, M, V]
        probs = tf.math.softmax(weights, axis=-1)
        weighted_emb = tf.matmul(probs, emb_vector)  # [B, M, H]
        weighted_emb = weighted_emb - tf.reshape(zero_vector, [1, 1, -1])
        weighted_emb = weighted_emb * mask
        return weighted_emb


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

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.params.vocab_size,
            output_dim=embedding_size,
            mask_zero=self.params.mask_zero,
            name="word_embeddings"
        )
        if self.params.extra_tokens_vocab_size is not None:
            self.extra_word_embeddings_layer = MyExtraEmbedding(
                input_dim=self.params.extra_tokens_vocab_size + 1,  # +1 is for a <pad>/0 vector
                output_dim=embedding_size,
                mask_zero=self.params.mask_zero,
                embeddings_initializer=self.create_initializer(),
                name="extra_word_embeddings",
            )
            self.extra_word_embeddings_layer.set_target_emb(self.word_embeddings_layer)

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
