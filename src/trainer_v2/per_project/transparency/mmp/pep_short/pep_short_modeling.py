from dataclasses import dataclass

import tensorflow as tf

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, BERT_CLS, define_bert_input
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF


def define_inputs(max_seq_len) -> dict[str, tf.keras.layers.Input]:
    inputs_d = {}
    for i in [1, 2]:
        inputs_d[f"input_ids{i}"] = tf.keras.layers.Input(
            shape=(max_seq_len,), dtype='int32', name=f"input_ids{i}")
        inputs_d[f"segment_ids{i}"] = tf.keras.layers.Input(
            shape=(max_seq_len,), dtype='int32', name=f"segment_ids{i}")
        inputs_d[f"s{i}"] = tf.keras.layers.Input(
            shape=(), dtype='float32', name=f"s{i}")
    return inputs_d

@dataclass
class PEPShortModelConfig(ModelConfigType):
    max_seq_length = 16
    num_classes = 1


def set_value(var, new_val):
    # Placeholder for the new value
    new_value = tf.compat.v1.placeholder(tf.int32)

    # Operation to update the variable
    update_var = tf.compat.v1.assign(var, new_value)

    # Initialize the variables
    init = tf.compat.v1.global_variables_initializer()

    # Create a session
    with tf.compat.v1.Session() as sess:
        # Run the initializer
        sess.run(init)

        # Update the variable
        sess.run(update_var, feed_dict={new_value: new_val})
        # Fetch the updated value
        print(sess.run(var))


class PEP_TTShort(ModelV2IF):
    def __init__(self, model_config: ModelConfigType):
        self.model_config = model_config
        super(PEP_TTShort, self).__init__()

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

        pairwise_model = self.build_pairwise_train_network()
        self.model: tf.keras.Model = pairwise_model

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
        pairwise_model = self.build_pairwise_train_network()
        self.inf_model = self.define_pointwise_model()
        self.model: tf.keras.Model = pairwise_model

    def define_pointwise_model(self):
        l_input_ids, l_token_type_ids = define_bert_input(
            self.model_config.max_seq_length, "")
        # [batch_size, dim]
        feature_rep = self.bert_cls.apply([l_input_ids, l_token_type_ids])
        hidden = self.dense1(feature_rep)
        z1 = self.dense2(hidden)

        inputs = (l_input_ids, l_token_type_ids)
        model = tf.keras.Model(inputs=inputs, outputs=z1, name="bert_model")
        return model

    def build_pairwise_train_network(self):
        l_input_ids1, l_token_type_ids1 = define_bert_input(
            self.model_config.max_seq_length, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(
            self.model_config.max_seq_length, "2")
        s1 = tf.keras.layers.Input(shape=(), dtype='float32', name=f"s1")
        s2 = tf.keras.layers.Input(shape=(), dtype='float32', name=f"s2")
        # inputs = [l_input_ids1, l_token_type_ids1, s1, l_input_ids2, l_token_type_ids2, s2]
        variable_list = [l_input_ids1, l_token_type_ids1, s1,
                         l_input_ids2, l_token_type_ids2, s2]
        variable_names = ["input_ids1", "segment_ids1", "s1",
                          "input_ids2", "segment_ids2", "s2"]

        inputs = {name: value for name, value in zip(variable_names, variable_list)}

        def apply(inputs):
            feature_rep = self.bert_cls.apply(inputs)
            hidden = self.dense1(feature_rep)
            return self.dense2(hidden)

        l_input_ids = tf.concat([l_input_ids1, l_input_ids2], axis=0)
        l_token_type_ids = tf.concat([l_token_type_ids1, l_token_type_ids2], axis=0)

        z_concat = apply([l_input_ids, l_token_type_ids])

        B, _ = get_shape_list2(l_input_ids1)
        z_stack = tf.reshape(z_concat, [2, B, -1])
        z1 = z_stack[0]
        z2 = z_stack[1]
        score_stack = tf.transpose(z_stack, [1, 0, 2])
        # tf.reshape()
        #
        # z1 = apply([l_input_ids1, l_token_type_ids1])
        # z2 = apply([l_input_ids2, l_token_type_ids2])
        #
        # score_stack = tf.stack([z1, z2], axis=1)

        losses = self.distilation_loss(s1, s2, z1, z2)
        loss = tf.reduce_mean(losses)
        outputs = score_stack, loss
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bert_model")
        return model

    def distilation_loss(self, s1, s2, z1, z2):
        margin1 = s1 - s2
        margin2 = z1 - z2
        losses = tf.math.square(margin1 - margin2)
        return losses

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        checkpoint = tf.train.Checkpoint(self.model)
        checkpoint.restore(init_checkpoint)
        self.model.optimizer.iterations.assign(0)
