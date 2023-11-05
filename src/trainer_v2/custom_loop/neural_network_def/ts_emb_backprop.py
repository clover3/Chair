from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, BERT_CLS, define_bert_input
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.per_project.transparency.mmp.pep.bert_embedding_backprop import CustomBertModelLayer

class EmbTrainIF(TwoSegConcatLogitCombineTwoModel):
    @abstractmethod
    def get_target_rep(self, neg_spe_emb_idx):
        pass



class TSEmbBackprop(EmbTrainIF):
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

        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)
        self.comb_layer = self.combine_local_decisions_layer()
        self.emb_learning: keras.Model = self.define_emb_learning()

        checkpoint = tf.train.Checkpoint(self.emb_learning)
        checkpoint.restore(run_config.train_config.init_checkpoint).expect_partial()

    def get_extra_embedding_layer(self):
        return self.l_bert.embeddings_layer.extra_word_embeddings_layer

    def get_word_embedding_layer(self):
        return self.l_bert.embeddings_layer.word_embeddings_layer

    def define_emb_learning(self):
        # Difference from
        segment_len = self.max_seq_length // 2
        l_input_ids, l_token_type_ids = define_bert_input(segment_len, "")
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = self.bert_cls.apply(inputs)
        B, _ = get_shape_list2(l_input_ids)
        #  [batch_size, num_window, dim2 ]
        hidden = self.dense1(feature_rep)
        local_decisions = self.dense2(hidden)

        predictions = local_decisions
        losses = -tf.abs(predictions)  # Higher scores are better

        loss = tf.reduce_mean(losses)
        outputs = [predictions, loss]
        # [batch_size, dim]
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=outputs, name="bert_model")

        layers_to_set_not_trainble = [
            self.pooler, self.dense1, self.dense2,
            self.l_bert.embeddings_layer.word_embeddings_layer,
            self.l_bert.embeddings_layer.position_embeddings_layer,
            self.l_bert.embeddings_layer.token_type_embeddings_layer,
            self.l_bert.embeddings_layer.layer_norm_layer,
            self.l_bert.encoders_layer
        ]
        for layer in layers_to_set_not_trainble:
            layer.trainable = False

        return model

    def recursive_set_not_trainable(self, cur_layer):
        extra_emb = self.get_extra_embedding_layer()
        target_name = extra_emb.name
        print("target_name", target_name)
        print(f"Current layer {cur_layer.name}")
        if cur_layer.name == target_name:
            cur_layer.trainable = True
            contain_target = True
        else:
            if hasattr(cur_layer, 'layers'):  # if it's a nested model or layer
                contain_target = False
                for child_layer in cur_layer.layers:
                    print(f"child_layer {child_layer.name}")
                    ret = self.recursive_set_not_trainable(child_layer)
                    contain_target = contain_target or ret
            else:
                contain_target = False

        if not contain_target:
            print(f"set {cur_layer.name} to not trainable")
            cur_layer.trainable = False

        return contain_target

    def get_keras_model(self):
        return self.emb_learning

    def get_target_rep(self, neg_spe_emb_idx):
        extra_emb_layer: tf.keras.layers.Embedding = self.get_extra_embedding_layer()
        def get_row_of_embedding(emb_layer, i) -> np.array:
            embedding_weights = emb_layer.get_weights()[0]
            ith_embedding = embedding_weights[i]
            return ith_embedding

        target_emb = get_row_of_embedding(extra_emb_layer, abs(neg_spe_emb_idx))
        return target_emb


class TSEmbWeights(TSEmbBackprop):
    def build_model(self, run_config):
        bert_params = load_bert_config(get_bert_config_path())
        bert_params.extra_tokens_vocab_size = 10
        self.num_window = 2
        prefix = "encoder"
        self.num_classes = self.model_config.num_classes
        self.max_seq_length = self.model_config.max_seq_length
        self.window_length = int(self.max_seq_length / self.num_window)

        self.l_bert = CustomBertModelLayer.from_params(
            bert_params, name="{}/bert".format(prefix))
        self.pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))

        self.bert_cls = BERT_CLS(self.l_bert, self.pooler)
        self.dense1 = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)
        self.comb_layer = self.combine_local_decisions_layer()
        self.emb_learning: keras.Model = self.define_emb_learning()

        checkpoint = tf.train.Checkpoint(self.emb_learning)
        checkpoint.restore(run_config.train_config.init_checkpoint).expect_partial()

    def get_target_rep(self, neg_spe_emb_idx):
        extra_emb_layer: tf.keras.layers.Embedding = self.get_extra_embedding_layer()
        input_ids_like = tf.constant([-neg_spe_emb_idx], tf.int32)
        input_ids_like = tf.reshape(input_ids_like, [1, 1])
        emb_out = extra_emb_layer(input_ids_like)  # [1, 1, H]

        return tf.reshape(emb_out, [-1])

