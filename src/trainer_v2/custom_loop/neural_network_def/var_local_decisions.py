import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import load_stock_weights, define_bert_input, ModelConfig, \
    BERT_CLS, load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack


# Num Local Config =4
class NLC(ModelConfigType):
    max_seq_length = 600
    num_classes = 3
    num_local_classes = 3


class NLC4(NLC):
    max_seq_length = 600
    num_classes = 3
    num_local_classes = 4


class TSCVarLD(ClassificationModelIF):
    # TSC: Two Segment Concat
    # VarLD: Variable Local decisions
    def __init__(self, combine_ld):
        super(TSCVarLD, self).__init__()
        self.combine_ld = combine_ld

    def build_model(self, bert_params, config: NLC):
        num_window = 2
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        num_local_classes = config.num_local_classes
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")

        # [batch_size, dim]
        window_length = int(max_seq_length / num_window)
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(bert_cls.apply, inputs,
                                                       max_seq_length, window_length)

        B, _ = get_shape_list2(l_input_ids)
        # [batch_size, num_window, dim2 ]
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_local_classes, activation=tf.nn.softmax)(hidden)
        comb_layer = self.combine_ld()
        output = comb_layer(local_decisions)
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls
        self.l_bert = l_bert
        self.pooler = pooler

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)


def keep_if_0_or_m(input_ids, segment_ids, m):
    is_seg1 = tf.equal(segment_ids, 0)
    is_seg2_a = tf.equal(segment_ids, m)
    is_ids_kept = tf.logical_or(is_seg1, is_seg2_a)
    keep_mask = tf.cast(is_ids_kept, tf.int32)  # [1 if keep
    input_ids = input_ids * keep_mask
    return input_ids


class SingleVarLD(ClassificationModelIF):
    def __init__(self, combine_ld):
        super(SingleVarLD, self).__init__()
        self.combine_ld = combine_ld

    def build_model(self, bert_params, config: NLC):
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        num_local_classes = config.num_local_classes
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")

        l_input_ids_a = keep_if_0_or_m(l_input_ids, l_token_type_ids, 1)
        l_input_ids_b = keep_if_0_or_m(l_input_ids, l_token_type_ids, 2)

        input_ids = tf.concat([l_input_ids_a, l_input_ids_b], axis=0)
        segment_ids = tf.concat([l_token_type_ids, l_token_type_ids], axis=0)
        segment_ids = tf.minimum(segment_ids, 1)  # Change 2 to 1

        feature_rep_flat = bert_cls.apply([input_ids, segment_ids])

        B, _ = get_shape_list2(l_input_ids)
        feature_rep = tf.reshape(feature_rep_flat, [B, 2, bert_params.hidden_size])
        # [batch_size, num_window, dim2 ]
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_local_classes, activation=tf.nn.softmax)(hidden)
        comb_layer = self.combine_ld()
        output = comb_layer(local_decisions)
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls
        self.l_bert = l_bert
        self.pooler = pooler

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)
