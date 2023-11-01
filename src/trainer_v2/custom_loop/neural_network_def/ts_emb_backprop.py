import tensorflow as tf
from tensorflow import keras

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, BERT_CLS, define_bert_input
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel


class TSEmbBackprop(TwoSegConcatLogitCombineTwoModel):
    def build_model(self, run_config):
        bert_params = load_bert_config(get_bert_config_path())
        bert_params.extra_tokens_vocab_size = 10
        self.num_window = 2
        prefix = "encoder"
        self.num_classes = self.model_config.num_classes
        self.max_seq_length = self.model_config.max_seq_length
        self.window_length = int(self.max_seq_length / self.num_window)

        self.l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        self.extra_embedding = self.l_bert.embeddings_layer.extra_word_embeddings_layer
        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)
        self.comb_layer = self.combine_local_decisions_layer()

        self.emb_learning: keras.Model = self.define_emb_learning()

    def define_emb_learning(self):
        # Difference from
        segment_len = (self.max_seq_length / 2)
        l_input_ids, l_token_type_ids = define_bert_input(segment_len, "")
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = self.bert_cls.apply(inputs)
        B, _ = get_shape_list2(l_input_ids)
        #  [batch_size, num_window, dim2 ]
        hidden = self.dense1(feature_rep)
        local_decisions = self.dense2(hidden)

        # [batch_size, dim]
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=local_decisions, name="bert_model")
        return model