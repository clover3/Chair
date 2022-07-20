import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_stock_weights, define_bert_input, ModelConfig, \
    BERT_CLS, load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.asymmetric import BERTAsymmetricProjectMean
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack, \
    combine_local_decision_by_fuzzy_logic


class AsymmetricMeanPool(ClassificationModelIF):
    def __init__(self, inner_network=BERTAsymmetricProjectMean):
        super(AsymmetricMeanPool, self).__init__()
        self.inner_network = inner_network

    def build_model(self, bert_params, model_config):
        network = self.inner_network(bert_params, model_config)
        self.network = network

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.network.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights(l_bert2, init_checkpoint, n_expected_restore=197)


class TwoSegConcat(ClassificationModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(TwoSegConcat, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig):
        num_window = 2
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")

        # [batch_size, dim]
        window_length = int(max_seq_length / num_window)
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(bert_cls.apply, inputs,
                                                       max_seq_length, window_length)

        # [batch_size, num_window, dim2 ]
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        self.local_decisions = local_decisions
        combine_local_decisions = self.combine_local_decisions_layer()
        self.cld = combine_local_decisions
        output = combine_local_decisions(local_decisions)
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


class TwoSegConcat2(ClassificationModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(TwoSegConcat2, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig):
        num_window = 2
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")

        # [batch_size, dim]
        window_length = int(max_seq_length / num_window)
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(bert_cls.apply, inputs,
                                                       max_seq_length, window_length)

        B, _ = get_shape_list2(l_input_ids)
        combine_mask = tf.ones([B, 2], tf.int32)
        # [batch_size, num_window, dim2 ]
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        output = combine_local_decision_by_fuzzy_logic(local_decisions)
        # output = local_decisions[:, 0]
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
