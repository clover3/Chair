import tensorflow as tf
from tensorflow import keras

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS, define_bert_input, load_bert_config, \
    load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF


class TSConcatDistil(ModelV2IF):
    def __init__(self, model_config, combine_local_decisions_layer):
        super(TSConcatDistil, self).__init__()
        self.model_config = model_config
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, _run_config):
        bert_params = load_bert_config(get_bert_config_path())
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

        # self.point_model: keras.Model = self.define_pointwise_model()
        self.pair_model: keras.Model = self.define_pairwise_train_model()
        self.model = self.pair_model

    def define_pointwise_model(self):
        l_input_ids, l_token_type_ids = define_bert_input(self.max_seq_length, "")
        # [batch_size, dim]
        output = self.apply_predictor(l_input_ids, l_token_type_ids)
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        return model

    def define_pairwise_train_model(self):
        max_seq_length = self.max_seq_length
        l_input_ids1, l_token_type_ids1 = define_bert_input(max_seq_length, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(max_seq_length, "2")
        teacher_score1 = keras.layers.Input(shape=(1,), dtype='float32', name="score1")
        teacher_score2 = keras.layers.Input(shape=(1,), dtype='float32', name="score2")

        # [batch_size, dim]
        inputs = [l_input_ids1, l_token_type_ids1, teacher_score1,
                  l_input_ids2, l_token_type_ids2, teacher_score2]
        output1 = self.apply_predictor(l_input_ids1, l_token_type_ids1)
        output2 = self.apply_predictor(l_input_ids2, l_token_type_ids2)
        both_output = tf.stack([output1, output2], axis=1)

        losses = self.loss_fn(output1, output2, teacher_score1, teacher_score2)
        loss = tf.reduce_mean(losses)

        outputs = [both_output, loss]
        model = keras.Model(inputs=inputs, outputs=outputs, name="bert_model")
        return model

    def loss_fn(self, pos_score, neg_score, teacher_score1, teacher_score2):
        margin_student = pos_score - neg_score
        margin_teacher = teacher_score1 - teacher_score2
        loss = tf.math.square(margin_student - margin_teacher)
        return loss

    def apply_predictor(self, l_input_ids, l_token_type_ids):
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(
            self.bert_cls.apply, inputs,
            self.max_seq_length, self.window_length)
        B, _ = get_shape_list2(l_input_ids)
        hidden = self.dense1(feature_rep)
        local_decisions = self.dense2(hidden)
        output = self.comb_layer(local_decisions)
        return output

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)



