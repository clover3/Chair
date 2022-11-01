from trainer_v2.bert_for_tf2.spit_attn_probs.bert_layer import BertModelLayerSAP, BertClsSAP
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_checkpoint, define_bert_input
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
import tensorflow as tf


class BertSAP(ClassificationModelIF):
    def build_model(self, bert_params, model_config):
        l_bert = BertModelLayerSAP.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        bert_cls = BertClsSAP(l_bert, pooler)
        max_seq_len = model_config.max_seq_length
        num_classes = model_config.num_classes
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_len, "")
        pooled = bert_cls.apply([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
        model = tf.keras.Model(inputs=(l_input_ids, l_token_type_ids), outputs=output, name="bert_model")
        self.model: tf.keras.Model = model
        self.bert_cls = bert_cls
        self.l_bert = l_bert
        self.pooler = pooler

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        return load_bert_checkpoint(self.bert_cls, init_checkpoint)




