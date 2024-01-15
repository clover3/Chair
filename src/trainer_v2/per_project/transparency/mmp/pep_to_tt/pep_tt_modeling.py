import dataclasses

import tensorflow as tf

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, BERT_CLS, define_bert_input, \
    load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF


@dataclasses.dataclass
class PEP_TT_ModelConfig(ModelConfigType):
    max_seq_length = 16
    num_classes = 1
    max_num_terms = 10


def define_inputs(max_term_cnt, max_seq_len) -> dict[str, tf.keras.layers.Input]:
    inputs_d = {}
    for role in ["pos", "neg"]:
        inputs_d[f"{role}_input_ids"] = tf.keras.layers.Input(
            shape=(max_term_cnt, max_seq_len,), dtype='int32', name=f"{role}_input_ids")
        inputs_d[f"{role}_segment_ids"] = tf.keras.layers.Input(
            shape=(max_term_cnt, max_seq_len,), dtype='int32', name=f"{role}_segment_ids")
        inputs_d[f"{role}_multiplier_arr"] = tf.keras.layers.Input(
            shape=(max_term_cnt,), dtype='float32', name=f"{role}_multiplier_arr")
        inputs_d[f"{role}_norm_add_factor"] = tf.keras.layers.Input(
            shape=(), dtype='float32', name=f"{role}_norm_add_factor")
        inputs_d[f"{role}_value_score"] = tf.keras.layers.Input(
            shape=(), dtype='float32', name=f"{role}_value_score")
    return inputs_d


def define_inputs_single(max_seq_len) -> dict[str, tf.keras.layers.Input]:
    inputs_d = {}
    for role in ["pos", "neg"]:
        inputs_d[f"{role}_input_ids"] = tf.keras.layers.Input(
            shape=(max_seq_len,), dtype='int32', name=f"{role}_input_ids")
        inputs_d[f"{role}_segment_ids"] = tf.keras.layers.Input(
            shape=(max_seq_len,), dtype='int32', name=f"{role}_segment_ids")
        inputs_d[f"{role}_multiplier"] = tf.keras.layers.Input(
            shape=(), dtype='float32', name=f"{role}_multiplier")
        inputs_d[f"{role}_norm_add_factor"] = tf.keras.layers.Input(
            shape=(), dtype='float32', name=f"{role}_norm_add_factor")
        inputs_d[f"{role}_value_score"] = tf.keras.layers.Input(
            shape=(), dtype='float32', name=f"{role}_value_score")
    return inputs_d


def weighted_sum(scores):
    weights = tf.nn.softmax(scores, axis=1)
    return tf.reduce_sum(scores * weights, axis=1)


class PEP_TT_Model(ModelV2IF):
    def __init__(self, model_config: PEP_TT_ModelConfig):
        self.model_config = model_config
        super(PEP_TT_Model, self).__init__()

    def build_model(self, _):
        prefix = "encoder"
        bert_params = load_bert_config(get_bert_config_path())
        self.max_seq_length = self.model_config.max_seq_length
        self.l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        self.comb_layer = CombineByScoreAdd()

        self.pointwise_src = self.define_pointwise_model()
        self.build_pairwise_train_network()

    def build_model_for_inf(self, _):
        prefix = "encoder"
        bert_params = load_bert_config(get_bert_config_path())
        self.max_seq_length = self.model_config.max_seq_length
        self.l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

        self.build_pairwise_train_network()
        self.inf_model = self.build_term_pair_scoring_network()

    def apply_predictor(self, l_input_ids, l_token_type_ids):
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(
            self.bert_cls.apply, inputs,
            32, 16)
        B, _ = get_shape_list2(l_input_ids)
        # [batch_size, num_window, dim2 ]
        hidden = self.dense1(feature_rep)
        local_decisions = self.dense2(hidden)
        output = self.comb_layer(local_decisions)
        return output

    def define_pointwise_model(self):
        l_input_ids, l_token_type_ids = define_bert_input(32, "")
        # [batch_size, dim]
        output = self.apply_predictor(l_input_ids, l_token_type_ids)
        inputs = (l_input_ids, l_token_type_ids)
        model = tf.keras.Model(inputs=inputs, outputs=output, name="bert_model")
        return model

    def build_pairwise_train_network(self):
        max_num_terms = self.model_config.max_num_terms
        max_seq_length = self.model_config.max_seq_length
        inputs_d = define_inputs(max_num_terms, max_seq_length)
        # [batch_size, dim]
        def flatten3_2(t):
            return tf.reshape(t, [-1, max_seq_length])

        score_d = {}
        for role in ["pos", "neg"]:
            input_ids = flatten3_2(inputs_d[f"{role}_input_ids"])
            segment_ids = flatten3_2(inputs_d[f"{role}_segment_ids"])
            feature_rep = self.bert_cls.apply([input_ids, segment_ids])
            hidden = self.dense1(feature_rep)
            pep_pred = self.dense2(hidden)
            probs_flat = tf.nn.sigmoid(pep_pred)
            probs = tf.reshape(probs_flat, [-1, max_num_terms, 1])  # [B, M, 1]
            norm_add_factor = inputs_d[f"{role}_norm_add_factor"]
            def add_two_dims(t):
                t = tf.expand_dims(t, axis=1)
                t = tf.expand_dims(t, axis=1)
                return t
            nom = probs + add_two_dims(norm_add_factor) + 1e-8 # [B, M, 1]
            denom = probs * tf.expand_dims(inputs_d[f"{role}_multiplier_arr"], axis=2)  # [B, 1, 1]
            new_align_score = denom / nom
            new_align_score = weighted_sum(new_align_score)

            value_score = inputs_d[f"{role}_value_score"]
            total_score = new_align_score + tf.expand_dims(value_score, axis=1)
            score_d[role] = total_score

        score_stack = tf.stack([score_d["pos"], score_d["neg"]], axis=1)
        losses = tf.maximum(1 - (score_d["pos"] - score_d["neg"]), 0)
        loss = tf.reduce_mean(losses)
        outputs = score_stack, loss
        model = tf.keras.Model(inputs=inputs_d, outputs=outputs, name="bert_model")
        self.model: tf.keras.Model = model

    def build_term_pair_scoring_network(self):
        max_num_terms = self.model_config.max_num_terms
        max_seq_length = self.model_config.max_seq_length
        input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name=f"input_ids")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name=f"segment_ids")

        feature_rep = self.bert_cls.apply([input_ids, segment_ids])
        hidden = self.dense1(feature_rep)
        pep_pred = self.dense2(hidden)
        probs = tf.nn.sigmoid(pep_pred)
        inputs = [input_ids, segment_ids]
        outputs = [probs, ]
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="bert_model")

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        checkpoint = tf.train.Checkpoint(self.pointwise_src)
        checkpoint.restore(init_checkpoint)


def bm25_like(probs, multiplier, norm_add_factor, value_score):
    nom = probs + tf.expand_dims(norm_add_factor, axis=1) + 1e-8  # [B, 1]
    denom = probs * tf.expand_dims(multiplier, axis=1)  # [B, 1]
    new_align_score = denom / nom
    total_score = new_align_score + tf.expand_dims(value_score, axis=1)
    return total_score


class PEP_TT_Model_Single(PEP_TT_Model):
    def build_pairwise_train_network(self):
        max_seq_length = self.model_config.max_seq_length
        inputs_d = define_inputs_single(max_seq_length)
        # [batch_size, dim]

        score_d = {}
        for role in ["pos", "neg"]:
            input_ids = inputs_d[f"{role}_input_ids"]
            segment_ids = inputs_d[f"{role}_segment_ids"]
            feature_rep = self.bert_cls.apply([input_ids, segment_ids])
            hidden = self.dense1(feature_rep)
            pep_pred = self.dense2(hidden)
            probs = tf.nn.sigmoid(pep_pred)
            norm_add_factor = inputs_d[f"{role}_norm_add_factor"]
            multiplier = inputs_d[f"{role}_multiplier"]
            value_score = inputs_d[f"{role}_value_score"]

            total_score = bm25_like(probs, multiplier, norm_add_factor, value_score)
            score_d[role] = total_score

        score_stack = tf.stack([score_d["pos"], score_d["neg"]], axis=1)
        hinge_losses = tf.maximum(1 - (score_d["pos"] - score_d["neg"]), 0)
        loss = tf.reduce_mean(hinge_losses)
        outputs = score_stack, loss
        model = tf.keras.Model(inputs=inputs_d, outputs=outputs, name="bert_model")
        self.model: tf.keras.Model = model


class PEP_TT_Model_Single2(PEP_TT_Model):
    def build_pairwise_train_network(self):
        max_seq_length = self.model_config.max_seq_length
        inputs_d = define_inputs_single(max_seq_length)
        # [batch_size, dim]

        score_d = {}
        for role in ["pos", "neg"]:
            input_ids = inputs_d[f"{role}_input_ids"]
            segment_ids = inputs_d[f"{role}_segment_ids"]
            feature_rep = self.bert_cls.apply([input_ids, segment_ids])
            hidden = self.dense1(feature_rep)
            pep_pred = self.dense2(hidden)
            probs = tf.nn.sigmoid(pep_pred)
            norm_add_factor = inputs_d[f"{role}_norm_add_factor"]
            multiplier = inputs_d[f"{role}_multiplier"]
            value_score = inputs_d[f"{role}_value_score"]

            total_score = bm25_like(probs, multiplier, norm_add_factor, value_score)
            score_d[role] = total_score

        score_stack = tf.stack([score_d["pos"], score_d["neg"]], axis=1)
        hinge_losses = tf.maximum(1 - (score_d["pos"] - score_d["neg"]), 0)
        loss = tf.reduce_mean(hinge_losses)

        avg_output = tf.reduce_mean(score_stack)
        reg_loss = avg_output * 0.2
        loss = loss + reg_loss
        outputs = score_stack, loss
        model = tf.keras.Model(inputs=inputs_d, outputs=outputs, name="bert_model")
        self.model: tf.keras.Model = model


class PEP_TT_Model_Single_BERT_Init(PEP_TT_Model_Single):
    def build_model(self, _):
        prefix = "encoder"
        bert_params = load_bert_config(get_bert_config_path())
        self.max_seq_length = self.model_config.max_seq_length
        self.l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        self.comb_layer = CombineByScoreAdd()
        self.build_pairwise_train_network()

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)
