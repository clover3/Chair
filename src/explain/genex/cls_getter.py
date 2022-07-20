import tensorflow as tf
from tensorflow import keras

from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_checkpoint, BERT_CLS


class CLSPooler:
    def build_model(self, bert_params):
        max_seq_len1 = 512
        l_input_ids = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids")
        # l_token_type_ids = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids")
        l_token_type_ids= tf.zeros_like(l_input_ids, tf.int32)
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        self.bert_cls = BERT_CLS(l_bert, pooler)
        cls = self.bert_cls.apply([l_input_ids, l_token_type_ids])

        inputs = (l_input_ids,)
        model = keras.Model(inputs=inputs, outputs=[cls], name="bert_model")
        self.model: keras.Model = model

    def init_checkpoint(self, bert_checkpoint):
        load_bert_checkpoint(self.bert_cls, bert_checkpoint)
