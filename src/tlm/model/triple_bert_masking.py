from typing import Dict

import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel, create_initializer
from tlm.model.dual_model_common import *


class TripleBertMasking(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(TripleBertMasking, self).__init__()

        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        input_ids3 = features["input_ids3"]
        input_mask3 = features["input_mask3"]
        segment_ids3 = features["segment_ids3"]

        with tf.compat.v1.variable_scope(triple_model_prefix1):
            model_1 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(triple_model_prefix2):
            model_2 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids2,
                input_mask=input_mask2,
                token_type_ids=segment_ids2,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(triple_model_prefix3):
            model_3 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids3,
                input_mask=input_mask3,
                token_type_ids=segment_ids3,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]
        model_2_first_token = model_2.get_sequence_output()[:, 0, :]

        pooled3 = model_3.get_pooled_output()
        probs3 = tf.keras.layers.Dense(2,
                                       activation=tf.keras.activations.softmax,
                                       kernel_initializer=create_initializer(config.initializer_range))(pooled3)
        mask_scalar = probs3[:, 1:2]
        self.rel_score = mask_scalar

        model_2_first_token = mask_scalar * model_2_first_token

        rep = tf.concat([model_1_first_token, model_2_first_token], axis=1)

        self.sequence_output = tf.concat([model_1.get_sequence_output(), model_2.get_sequence_output()], axis=2)
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        pooled_output = dense_layer(rep)
        self.pooled_output = pooled_output

    def get_predictions(self) -> Dict:
        return {
            'rel_score': self.rel_score
        }

    def get_trainable_vars_for_relevance_tuning(self):
        r = []
        for v in tf.compat.v1.trainable_variables():
            if v.name.startswith(triple_model_prefix1) or v.name.startswith(triple_model_prefix2):
                print("Skip: ", v.name)
            else:
                print("Trainable:", v.name)
                r.append(v)

        return r


class TripleBertMaskingRelevanceTuning(TripleBertMasking):
    def get_trainable_vars(self):
        r = []
        for v in tf.compat.v1.trainable_variables():
            if v.name.startswith(triple_model_prefix1) or v.name.startswith(triple_model_prefix2):
                print("Skip: ", v.name)
            else:
                print("Trainable:", v.name)
                r.append(v)

        return r


class TripleBertWeighted(BertModelInterface):
    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               features=None,
               scope=None):
        super(TripleBertWeighted, self).__init__()

        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        input_ids3 = features["input_ids3"]
        input_mask3 = features["input_mask3"]
        segment_ids3 = features["segment_ids3"]

        def apply_binary_dense(vector):
            output = tf.keras.layers.Dense(2,
                                           activation=tf.keras.activations.softmax,
                                           name="cls_dense",
                                           kernel_initializer=create_initializer(config.initializer_range))(vector)
            return output

        with tf.compat.v1.variable_scope(triple_model_prefix1):
            model_1 = BertModel(
                    config=config,
                    is_training=is_training,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )
            model_1_pred = tf.keras.layers.Dense(3,
                                           activation=tf.keras.activations.softmax,
                                           name="cls_dense",
                                           kernel_initializer=create_initializer(config.initializer_range))(model_1.get_pooled_output())
            model_1_pred = model_1_pred[:, :2]

        with tf.compat.v1.variable_scope(triple_model_prefix2):
            model_2 = BertModel(
                    config=config,
                    is_training=is_training,
                    input_ids=input_ids2,
                    input_mask=input_mask2,
                    token_type_ids=segment_ids2,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )
            model_2_pred = apply_binary_dense(model_2.get_pooled_output())

        with tf.compat.v1.variable_scope(triple_model_prefix3):
            model_3 = BertModel(
                    config=config,
                    is_training=is_training,
                    input_ids=input_ids3,
                    input_mask=input_mask3,
                    token_type_ids=segment_ids3,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )
            model_3_pred = apply_binary_dense(model_3.get_pooled_output())

        # Option : initialize dense

        combined_pred = model_1_pred * model_3_pred[:, 0:1] \
                        + model_2_pred * model_3_pred[:, 1:2]

        self.rel_score = model_3_pred[:, 1:2]
        self.pooled_output = combined_pred

    def get_predictions(self) -> Dict:
        return {
            'rel_score': self.rel_score
        }
