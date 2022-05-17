import tensorflow as tf
from official.modeling import tf_utils
from official.nlp.keras_nlp import layers

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.partial_processing.bert_encoder_layer import BertEncoderModule
from trainer_v2.partial_processing.network_utils import vector_three_feature


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
class SiameseClassifier(tf.keras.Model):
    def __init__(self, model_config, **kwargs):
        super(SiameseClassifier, self).__init__()
        bert_config = model_config.bert_config

        def build_keras_input(name):
            return tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=name)

        sl1, sl2 = model_config.max_seq_length_list
        pad_len = sl1 - sl2
        word_ids1 = build_keras_input('input_ids1')
        mask1 = build_keras_input('input_mask1')
        type_ids1 = build_keras_input('segment_ids1')

        word_ids2 = build_keras_input('input_ids2')
        mask2 = build_keras_input('input_mask2')
        type_ids2 = build_keras_input('segment_ids2')

        def pad(t):
            return tf.pad(t, [(0, 0), (0, pad_len)])

        word_ids = tf.concat([word_ids1, pad(word_ids2)], axis=0)
        mask = tf.concat([mask1, pad(mask2)], axis=0)
        type_ids = tf.concat([type_ids1, pad(type_ids2)], axis=0)

        inputs1 = (word_ids1, mask1, type_ids1)
        inputs2 = (word_ids2, mask2, type_ids2)

        sub_kwargs = dict(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            inner_dim=bert_config.intermediate_size,
            inner_activation=tf_utils.get_activation(bert_config.hidden_act),
            output_dropout=bert_config.hidden_dropout_prob,
            attention_dropout=bert_config.attention_probs_dropout_prob,
            max_sequence_length=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            embedding_width=bert_config.embedding_size,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range),
        )
        dense1 = tf.keras.layers.Dense(3, trainable=True)
        print("dense1", dense1, dense1.trainable_variables, dense1.trainable_weights)
        bert_encoder = BertEncoderModule(**sub_kwargs)
        self.bert_encoder = bert_encoder
        print("bert_encoder.trainable", bert_encoder.trainable_variables)

        inputs = (word_ids, mask, type_ids)
        cls_output = bert_encoder(inputs)
        # cls_output = outputs['pooled_output']

        batch_size, _ = get_shape_list2(word_ids1)
        cls_output1 = cls_output[:batch_size]
        cls_output2 = cls_output[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(model_config.bert_config.hidden_size, activation='relu')(feature_rep)
        predictions = tf.keras.layers.Dense(model_config.num_classes, name="sentence_prediction")(hidden)

        inputs = [inputs1, inputs2]

        super(SiameseClassifier, self).__init__(
            inputs=inputs, outputs=predictions, **kwargs)


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
class SiameseClassifier2(tf.keras.Model):
    def __init__(self, model_config, **kwargs):
        bert_config = model_config.bert_config

        def build_keras_input(name):
            return tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=name)

        sl1, sl2 = model_config.max_seq_length_list
        pad_len = sl1 - sl2
        word_ids1 = build_keras_input('input_ids1')
        mask1 = build_keras_input('input_mask1')
        type_ids1 = build_keras_input('segment_ids1')

        word_ids2 = build_keras_input('input_ids2')
        mask2 = build_keras_input('input_mask2')
        type_ids2 = build_keras_input('segment_ids2')

        def pad(t):
            return tf.pad(t, [(0, 0), (0, pad_len)])

        word_ids = tf.concat([word_ids1, pad(word_ids2)], axis=0)
        mask = tf.concat([mask1, pad(mask2)], axis=0)
        type_ids = tf.concat([type_ids1, pad(type_ids2)], axis=0)

        inputs1 = (word_ids1, mask1, type_ids1)
        inputs2 = (word_ids2, mask2, type_ids2)

        sub_kwargs = dict(
            vocab_size=bert_config.vocab_size,
            hidden_size=bert_config.hidden_size,
            num_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            inner_dim=bert_config.intermediate_size,
            inner_activation=tf_utils.get_activation(bert_config.hidden_act),
            output_dropout=bert_config.hidden_dropout_prob,
            attention_dropout=bert_config.attention_probs_dropout_prob,
            max_sequence_length=bert_config.max_position_embeddings,
            type_vocab_size=bert_config.type_vocab_size,
            embedding_width=bert_config.embedding_size,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range),
        )
        vocab_size = bert_config.vocab_size
        hidden_size = bert_config.hidden_size
        num_layers = bert_config.num_hidden_layers
        num_attention_heads = bert_config.num_attention_heads
        inner_dim = bert_config.intermediate_size
        inner_activation = tf_utils.get_activation(bert_config.hidden_act)
        output_dropout = bert_config.hidden_dropout_prob
        attention_dropout = bert_config.attention_probs_dropout_prob
        max_sequence_length = bert_config.max_position_embeddings
        type_vocab_size = bert_config.type_vocab_size
        embedding_width = bert_config.embedding_size
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range)
        output_range = None
        inputs = (word_ids, mask, type_ids)
        activation = tf_utils.get_activation(bert_config.hidden_act)
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
        # cls_output = outputs['pooled_output']

        batch_size, _ = get_shape_list2(word_ids1)
        cls_output1 = cls_output[:batch_size]
        cls_output2 = cls_output[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(model_config.bert_config.hidden_size, activation='relu')(feature_rep)
        predictions = tf.keras.layers.Dense(model_config.num_classes, name="sentence_prediction")(hidden)

        inputs = [inputs1, inputs2]

        super(SiameseClassifier2, self).__init__(
            inputs=inputs, outputs=predictions, **kwargs)

    def _build_embedding_layer(self):
        embedding_width = self._config_dict[
                              'embedding_width'] or self._config_dict['hidden_size']
        return layers.OnDeviceEmbedding(
            vocab_size=self._config_dict['vocab_size'],
            embedding_width=embedding_width,
            initializer=self._config_dict['initializer'],
            name='word_embeddings')
