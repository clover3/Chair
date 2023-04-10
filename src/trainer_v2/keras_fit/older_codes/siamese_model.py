import tensorflow as tf
from official.modeling import performance

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.custom_loop.modeling_common.network_utils import vector_three_feature
from trainer_v2.keras_fit.bert_encoder_module import get_bert_encoder_module
from trainer_v2.keras_fit.config_helper import MultiSegModelConfig
from trainer_v2.keras_fit.modeling import get_optimizer
from trainer_v2.run_config import RunConfigEx


class TwoSegmentInput:
    def __init__(self, max_seq_length_list):
        def build_keras_input(name):
            return tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=name)

        sl1, sl2 = max_seq_length_list
        pad_len = sl1 - sl2
        word_ids1 = build_keras_input('input_ids1')
        mask1 = build_keras_input('input_mask1')
        type_ids1 = build_keras_input('segment_ids1')

        word_ids2 = build_keras_input('input_ids2')
        mask2 = build_keras_input('input_mask2')
        type_ids2 = build_keras_input('segment_ids2')

        self.word_ids1 = word_ids1
        self.mask1 = mask1
        self.type_ids1 = type_ids1
        self.word_ids2 = word_ids2
        self.mask2 = mask2
        self.type_ids2 = type_ids2

        def pad(t):
            return tf.pad(t, [(0, 0), (0, pad_len)])
        self.word_ids = tf.concat([word_ids1, pad(word_ids2)], axis=0)
        self.mask = tf.concat([mask1, pad(mask2)], axis=0)
        self.type_ids = tf.concat([type_ids1, pad(type_ids2)], axis=0)

        batch_size, _ = get_shape_list2(word_ids1)
        self.batch_size = batch_size
        self.inputs1 = (word_ids1, mask1, type_ids1)
        self.inputs2 = (word_ids2, mask2, type_ids2)

    def get_batch_size(self):
        return self.batch_size

    def get_input_concat(self):
        return self.word_ids, self.mask, self.type_ids

    def get_keras_inputs(self):
        inputs1 = (self.word_ids1, self.mask1, self.type_ids1)
        inputs2 = (self.word_ids2, self.mask2, self.type_ids2)
        return [inputs1, inputs2]


def model_factory_siamese(model_config: MultiSegModelConfig, run_config: RunConfigEx):
    bert_config = model_config.bert_config

    def model_fn():
        ts_input = TwoSegmentInput(model_config.max_seq_length_list)
        bert_encoder = get_bert_encoder_module(bert_config)
        print("bert_encoder.trainable", bert_encoder.trainable_variables)

        cls_output = bert_encoder(ts_input.get_input_concat())
        batch_size = ts_input.get_batch_size()
        cls_output1 = cls_output[:batch_size]
        cls_output2 = cls_output[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(model_config.bert_config.hidden_size, activation='relu')(feature_rep)
        predictions = tf.keras.layers.Dense(model_config.num_classes, name="sentence_prediction")(hidden)

        outer_model = tf.keras.Model(
            inputs=ts_input.get_keras_inputs(),
            outputs=predictions)
        optimizer = get_optimizer(run_config)
        outer_model.optimizer = performance.configure_optimizer(optimizer)
        inner_model_list = [bert_encoder]
        return outer_model, inner_model_list
    return model_fn