import tensorflow as tf
from tensorflow import keras

from cpath import get_bert_config_path
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS, load_bert_checkpoint, \
    load_bert_config
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF


def define_bert_input(max_seq_len, post_fix=""):
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids{}".format(post_fix))
    l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids{}".format(post_fix))
    return l_input_ids, l_token_type_ids


class BERTConcatModel(ModelV2IF):
    def __init__(self, model_config):
        super(BERTConcatModel, self).__init__()
        self.model_config = model_config

    def build_model(self, _run_config):
        bert_params = load_bert_config(get_bert_config_path())
        self.num_window = 2
        prefix = "encoder"
        self.num_classes = self.model_config.num_classes
        self.max_seq_length = self.model_config.max_seq_length
        self.l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                            name="{}/bert/pooler/dense".format(prefix))

        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)

        self.point_model: keras.Model = self.define_pointwise_model()
        self.pair_model: keras.Model = self.define_pairwise_model()

    def define_pointwise_model(self):
        l_input_ids, l_token_type_ids = define_bert_input(self.max_seq_length, "")
        # [batch_size, dim]
        output = self.apply_predictor(l_input_ids, l_token_type_ids)
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        return model

    def apply_predictor(self, l_input_ids, l_token_type_ids):
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = self.bert_cls.apply(inputs)
        # [batch_size, dim2 ]
        hidden = self.dense1(feature_rep)
        output = self.dense2(hidden)
        return output

    def define_pairwise_model(self):
        max_seq_length = self.max_seq_length
        l_input_ids1, l_token_type_ids1 = define_bert_input(max_seq_length, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(max_seq_length, "2")

        # [batch_size, dim]
        inputs = [l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2]
        output1 = self.apply_predictor(l_input_ids1, l_token_type_ids1)
        output2 = self.apply_predictor(l_input_ids2, l_token_type_ids2)
        both_output = tf.stack([output1, output2], axis=1)
        loss = tf.reduce_mean(self.loss_fn(output1, output2))

        outputs = [both_output, loss]
        model = keras.Model(inputs=inputs, outputs=outputs, name="bert_model")
        return model

    def get_keras_model(self):
        return self.pair_model

    def loss_fn(self, pos_score, neg_score):
        loss = tf.maximum(1 - (pos_score - neg_score), 0)
        return loss

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)

    def load_checkpoint(self, model_save_path):
        checkpoint = tf.train.Checkpoint(self.pair_model)
        checkpoint.restore(model_save_path).expect_partial()
