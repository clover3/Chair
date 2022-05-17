import tensorflow as tf
from official.modeling import tf_utils, performance

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.partial_processing.bert_encoder_layer import BertEncoderModule
from trainer_v2.partial_processing.config_helper import MultiSegModelConfig
from trainer_v2.partial_processing.modeling import get_optimizer
from trainer_v2.partial_processing.network_utils import vector_three_feature
from trainer_v2.run_config import RunConfigEx


def model_factory_siamese(model_config: MultiSegModelConfig, run_config: RunConfigEx):
    bert_config = model_config.bert_config
    def model_fn():
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

        kwargs = dict(
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
            trainable=True,
        )
        dense1 = tf.keras.layers.Dense(3, trainable=True)
        print("dense1", dense1, dense1.trainable_variables, dense1.trainable_weights)
        bert_encoder = BertEncoderModule(**kwargs)
        print("bert_encoder.trainable", bert_encoder.trainable_variables)

        inputs = (word_ids, mask, type_ids)
        cls_output = bert_encoder(inputs)
        batch_size, _ = get_shape_list2(word_ids1)
        cls_output1 = cls_output[:batch_size]
        cls_output2 = cls_output[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(model_config.bert_config.hidden_size, activation='relu')(feature_rep)
        predictions = tf.keras.layers.Dense(model_config.num_classes, name="sentence_prediction")(hidden)

        inputs = [inputs1, inputs2]

        outer_model = tf.keras.Model(
            inputs=inputs,
            outputs=predictions)
        optimizer = get_optimizer(run_config)
        outer_model.optimizer = performance.configure_optimizer(optimizer)
        inner_model_list = [bert_encoder]
        return outer_model, inner_model_list
    return model_fn