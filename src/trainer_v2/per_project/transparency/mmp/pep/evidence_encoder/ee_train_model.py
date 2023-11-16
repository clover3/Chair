import abc

import tensorflow as tf
from tensorflow import keras
from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, BERT_CLS, define_bert_input_w_prefix, load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.ee_train_common import get_is_valid_mask


class EEModelConfigType:
    __metaclass__ = abc.ABCMeta
    segment_len = abc.abstractproperty()
    max_num_qt = abc.abstractproperty()
    max_num_dt = abc.abstractproperty()


class EEModelConfig32_2_20(EEModelConfigType):
    segment_len = 32
    max_num_qt = 2
    max_num_dt = 20


def get_scores_from_rep_dot_prod(q_rep, d_rep):
    score_arr = tf.reduce_sum(tf.expand_dims(q_rep, axis=2) * tf.expand_dims(d_rep, axis=1), axis=3)
    return score_arr


def loss_per_qd(pred_score_arr, label_scores_stack, is_valid_mask):
    """

    :param pred_score_arr: float shape=[B, n_qt, n_dt]
    :param is_valid_mask:  float shape=[B, n_qt, n_dt]
    :param label_scores_stack:  float shape=[B, n_qt, n_dt]
    :return:
    """
    bias = (1-is_valid_mask) * -1000000.0
    pred_score_arr = pred_score_arr + bias  # [B, n_qt, n_dt]
    pred_probs = tf.nn.softmax(pred_score_arr, axis=2)

    label_scores_stack = label_scores_stack + bias
    label_probs = tf.nn.softmax(label_scores_stack, axis=2)
    per_item_loss = -tf.reduce_sum(tf.reduce_sum(pred_probs * label_probs, axis=2), axis=1)
    return per_item_loss


def loss_abs(pred_score_arr, label_scores_stack, is_valid_mask):
    """

    :param pred_score_arr: float shape=[B, n_qt, n_dt]
    :param is_valid_mask:  float shape=[B, n_qt, n_dt]
    :param label_scores_stack:  float shape=[B, n_qt, n_dt]
    :return:
    """
    error = tf.abs(pred_score_arr - label_scores_stack)
    error = error * is_valid_mask
    per_item_loss = tf.reduce_sum(tf.reduce_sum(error, axis=2), axis=1)
    return per_item_loss


def loss_w_binary(pred_score_arr, label_scores_stack, is_valid_mask):
    """

    :param pred_score_arr: float shape=[B, n_qt, n_dt]
    :param is_valid_mask:  float shape=[B, n_qt, n_dt]
    :param label_scores_stack:  float shape=[B, n_qt, n_dt]
    :return:
    """
    bias = (1-is_valid_mask) * -1000000.0
    pred_score_arr = pred_score_arr + bias  # [B, n_qt, n_dt]
    pred_probs = tf.nn.softmax(pred_score_arr, axis=2)

    label_scores_stack = label_scores_stack + bias
    label_probs = tf.nn.softmax(label_scores_stack, axis=2)
    per_item_loss = -tf.reduce_sum(tf.reduce_sum(pred_probs * label_probs, axis=2), axis=1)
    return per_item_loss


def loss_batch(pred_score_arr, label_scores_stack, is_valid_mask):
    """

    :param pred_score_arr: float shape=[B, n_qt, n_dt]
    :param is_valid_mask:  float shape=[B, n_qt, n_dt]
    :param label_scores_stack:  float shape=[B, n_qt, n_dt]
    :return:
    """
    bias = (1-is_valid_mask) * -1e8
    pred_score_arr = pred_score_arr + bias  # [B, n_qt, n_dt]
    label_scores_stack = label_scores_stack + bias

    pred_flat = tf.reshape(pred_score_arr, [-1])
    label_scores_stack = tf.reshape(label_scores_stack, [-1])

    pred_probs = tf.nn.softmax(pred_flat)
    label_probs = tf.nn.softmax(label_scores_stack)
    batch_loss = -tf.reduce_sum(pred_probs * label_probs)
    return batch_loss


class EvidenceEncoder(ModelV2IF):
    def __init__(self, model_config: EEModelConfigType, get_loss_fn, label_type="float32"):
        super(EvidenceEncoder, self).__init__()
        self.model_config = model_config
        self.get_loss_fn = get_loss_fn
        self.label_type = label_type

    def build_model(self, _run_config):
        bert_params = load_bert_config(get_bert_config_path())
        self.num_window = 2
        prefix = "encoder"
        self.segment_len = self.model_config.segment_len
        self.l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size)
        self.train_model: keras.Model = self.define_train_model()

    def define_train_model(self):
        q_seq_len = self.model_config.max_num_qt * self.model_config.segment_len
        d_seq_len = self.model_config.max_num_dt * self.model_config.segment_len
        q_input_ids, q_token_type_ids = define_bert_input_w_prefix(q_seq_len, "q")
        d_input_ids, d_token_type_ids = define_bert_input_w_prefix(d_seq_len, "d")
        n_score_size = self.model_config.max_num_qt * self.model_config.max_num_dt
        label_scores = keras.layers.Input(shape=(n_score_size,), dtype=self.label_type, name="scores")
        label_scores_stack = tf.reshape(label_scores, [-1, self.model_config.max_num_qt, self.model_config.max_num_dt])

        # [batch_size, dim]
        q_rep = self.apply_encoder(q_input_ids, q_token_type_ids, q_seq_len)  # [batch_size, num_window, dim]
        d_rep = self.apply_encoder(d_input_ids, d_token_type_ids, d_seq_len)  #
        is_valid_mask = get_is_valid_mask(q_input_ids, d_input_ids, q_seq_len, d_seq_len, self.segment_len)
        pred_score_arr = get_scores_from_rep_dot_prod(q_rep, d_rep)  # [batch_size, max_num_qt, max_num_dt]

        losses = self.get_loss_fn(pred_score_arr, label_scores_stack, is_valid_mask)
        loss = tf.reduce_mean(losses)
        output = pred_score_arr, loss

        inputs = (q_input_ids, q_token_type_ids, d_input_ids, d_token_type_ids, label_scores)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        return model

    def apply_encoder(self, l_input_ids, l_token_type_ids, max_seq_len):
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(
            self.bert_cls.apply, inputs,
            max_seq_len, self.segment_len)
        B, _ = get_shape_list2(l_input_ids)

        # [batch_size, num_window, dim2 ]
        hidden = self.dense1(feature_rep)
        return hidden

    def get_keras_model(self):
        return self.train_model

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)


class AddLayer(tf.keras.layers.Layer):
    def __init__(self, ):
        super(AddLayer, self).__init__()
        self.dummy_param = tf.Variable(0.0, trainable=True)

    def call(self, inputs):
        # [batch, num_feature, 3]
        return inputs + self.dummy_param


class EvidenceEncoderBaseline(ModelV2IF):
    def __init__(self, model_config: EEModelConfigType, get_loss_fn, label_type="float32"):
        super(EvidenceEncoderBaseline, self).__init__()
        self.model_config = model_config
        self.get_loss_fn = get_loss_fn
        self.label_type = label_type
        self.const_val = 1.1

    def build_model(self, _run_config):
        self.segment_len = self.model_config.segment_len
        self.dummy_layer = AddLayer()
        self.train_model: keras.Model = self.define_train_model()


    def define_train_model(self):
        q_seq_len = self.model_config.max_num_qt * self.model_config.segment_len
        d_seq_len = self.model_config.max_num_dt * self.model_config.segment_len
        q_input_ids, q_token_type_ids = define_bert_input_w_prefix(q_seq_len, "q")
        d_input_ids, d_token_type_ids = define_bert_input_w_prefix(d_seq_len, "d")
        n_score_size = self.model_config.max_num_qt * self.model_config.max_num_dt
        label_scores = keras.layers.Input(shape=(n_score_size,), dtype=self.label_type, name="scores")
        label_scores_stack = tf.reshape(label_scores, [-1, self.model_config.max_num_qt, self.model_config.max_num_dt])

        # [batch_size, dim]
        pred_score_arr: tf.Tensor = tf.ones_like(label_scores_stack, tf.float32) * self.const_val  # [batch_size, max_num_qt, max_num_dt]
        pred_score_arr = self.dummy_layer(pred_score_arr)
        is_valid_mask = get_is_valid_mask(q_input_ids, d_input_ids, q_seq_len, d_seq_len, self.model_config.segment_len)
        losses = self.get_loss_fn(pred_score_arr, label_scores_stack, is_valid_mask)
        loss = tf.reduce_mean(losses)
        output = pred_score_arr, loss

        inputs = (q_input_ids, q_token_type_ids, d_input_ids, d_token_type_ids, label_scores)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        return model

    def get_keras_model(self):
        return self.train_model

    def init_checkpoint(self, init_checkpoint):
        pass


