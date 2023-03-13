import tensorflow as tf
from tensorflow import keras
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2.spit_attn_probs.bert_layer import BertModelLayerSAP, BertClsSAP
from trainer_v2.custom_loop.modeling_common.bert_common import ModelConfig300_3, define_bert_input, \
    load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack


class TwoSegConcat2SAP(BertBasedModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(TwoSegConcat2SAP, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig300_3):
        num_window = 2
        prefix = "encoder"
        l_bert = BertModelLayerSAP.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BertClsSAP(l_bert, pooler)
        num_classes = config.num_classes
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
        local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        comb_layer = self.combine_local_decisions_layer()
        output = comb_layer(local_decisions)
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
