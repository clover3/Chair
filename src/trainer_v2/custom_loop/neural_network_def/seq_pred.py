from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import define_bert_input, load_stock_weights
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack, \
    split_stack_flatten_encode_sequence
import tensorflow as tf
from tensorflow import keras


class SeqPrediction(BertBasedModelIF):
    def __init__(self):
        super(SeqPrediction, self).__init__()

    def build_model(self, bert_params, config: ModelConfigType):
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = l_bert(inputs)
        seq_prediction = tf.keras.layers.Dense(config.num_classes, activation=tf.nn.softmax)(feature_rep)
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=seq_prediction, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        load_stock_weights(self.l_bert, init_checkpoint, n_expected_restore=197)


class TwoSegSeqPrediction(BertBasedModelIF):
    def __init__(self):
        super(TwoSegSeqPrediction, self).__init__()

    def build_model(self, bert_params, config: ModelConfigType):
        num_window = 2
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")

        # [batch_size, dim]
        window_length = int(max_seq_length / num_window)
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_sequence(l_bert, inputs,
                                                          max_seq_length, window_length)

        # [batch_size, num_window, seq_length, dim2 ]
        seq_prediction = tf.keras.layers.Dense(config.num_classes, activation=tf.nn.softmax)(feature_rep)

        print("seq_prediction", seq_prediction.shape)

        # output = local_decisions[:, 0]
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=seq_prediction, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        load_stock_weights(self.l_bert, init_checkpoint, n_expected_restore=197)
