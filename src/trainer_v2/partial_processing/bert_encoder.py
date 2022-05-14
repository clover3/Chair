# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bert encoder network."""
# pylint: disable=g-classes-have-attributes

import tensorflow as tf

from official.nlp.keras_nlp import layers


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
class MyBertEncoder(tf.keras.Model):

    def __init__(
            self,
            word_ids,
            mask,
            type_ids,
            vocab_size,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            max_sequence_length=512,
            type_vocab_size=16,
            inner_dim=3072,
            inner_activation=lambda x: tf.keras.activations.gelu(x, approximate=True),
            output_dropout=0.1,
            attention_dropout=0.1,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            output_range=None,
            embedding_width=None,
            **kwargs):
        activation = tf.keras.activations.get(inner_activation)
        initializer = tf.keras.initializers.get(initializer)

        self._self_setattr_tracking = False
        self._config_dict = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'max_sequence_length': max_sequence_length,
            'type_vocab_size': type_vocab_size,
            'inner_dim': inner_dim,
            'inner_activation': tf.keras.activations.serialize(activation),
            'output_dropout': output_dropout,
            'attention_dropout': attention_dropout,
            'initializer': tf.keras.initializers.serialize(initializer),
            'output_range': output_range,
            'embedding_width': embedding_width,
        }

        if embedding_width is None:
            embedding_width = hidden_size
        self._embedding_layer = self._build_embedding_layer()
        word_embeddings = self._embedding_layer(word_ids)

        # Always uses dynamic slicing for simplicity.
        self._position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            max_length=max_sequence_length,
            name='position_embedding')
        position_embeddings = self._position_embedding_layer(word_embeddings)
        self._type_embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=embedding_width,
            initializer=initializer,
            use_one_hot=True,
            name='type_embeddings')
        type_embeddings = self._type_embedding_layer(type_ids)

        embeddings = tf.keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings])

        self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

        embeddings = self._embedding_norm_layer(embeddings)
        embeddings = (tf.keras.layers.Dropout(rate=output_dropout)(embeddings))

        # We project the 'embedding' output to 'hidden_size' if it is not already
        # 'hidden_size'.
        if embedding_width != hidden_size:
            self._embedding_projection = tf.keras.layers.experimental.EinsumDense(
                '...x,xy->...y',
                output_shape=hidden_size,
                bias_axes='y',
                kernel_initializer=initializer,
                name='embedding_projection')
            embeddings = self._embedding_projection(embeddings)

        self._transformer_layers = []
        data = embeddings
        attention_mask = layers.SelfAttentionMask()(data, mask)
        encoder_outputs = []
        for i in range(num_layers):
            if i == num_layers - 1 and output_range is not None:
                transformer_output_range = output_range
            else:
                transformer_output_range = None
            layer = layers.TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_dim=inner_dim,
                inner_activation=inner_activation,
                output_dropout=output_dropout,
                attention_dropout=attention_dropout,
                output_range=transformer_output_range,
                kernel_initializer=initializer,
                name='transformer/layer_%d' % i)
            self._transformer_layers.append(layer)
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        last_enocder_output = encoder_outputs[-1]
        # Applying a tf.slice op (through subscript notation) to a Keras tensor
        # like this will create a SliceOpLambda layer. This is better than a Lambda
        # layer with Python code, because that is fundamentally less portable.
        first_token_tensor = last_enocder_output[:, 0, :]
        self._pooler_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            kernel_initializer=initializer,
            name='pooler_transform')
        cls_output = self._pooler_layer(first_token_tensor)

        outputs = dict(
            sequence_output=encoder_outputs[-1],
            pooled_output=cls_output,
            encoder_outputs=encoder_outputs,
        )
        super(MyBertEncoder, self).__init__(
            inputs=[word_ids, mask, type_ids], outputs=outputs, **kwargs)

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def _build_embedding_layer(self):
        embedding_width = self._config_dict[
                              'embedding_width'] or self._config_dict['hidden_size']
        return layers.OnDeviceEmbedding(
            vocab_size=self._config_dict['vocab_size'],
            embedding_width=embedding_width,
            initializer=self._config_dict['initializer'],
            name='word_embeddings')

    def get_embedding_layer(self):
        return self._embedding_layer

    def get_config(self):
        return self._config_dict

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @property
    def pooler_layer(self):
        """The pooler dense layer after the transformer layers."""
        return self._pooler_layer

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
