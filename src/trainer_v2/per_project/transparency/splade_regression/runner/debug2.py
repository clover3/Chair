
import logging
import sys
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.combine_mat import MatrixCombine
from trainer_v2.custom_loop.neural_network_def.two_seg_concat import TwoSegConcat2
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.runner.concat.mat import ModelConfig
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset, get_dummy_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.trainer_vector_regression import TrainerVectorRegression
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import AutoTokenizer


def get_dummy_model(_):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    # w = tf.Variable(np.zeros([30522,], np.float32), dtype=tf.float32, trainable=True)
    #
    h = tf.cast(tf.reduce_sum(input_ids, axis=1, keepdims=True), tf.float32)
    output = tf.keras.layers.Dense(30522)(h)
    # output = tf.expand_dims(w, axis=0) * tf.expand_dims(h, axis=1)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[output])
    return new_model

import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_stock_weights, define_bert_input, ModelConfig300_3, \
    BERT_CLS, load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.asymmetric import BERTAsymmetricProjectMean
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.segmented_enc import split_stack_flatten_encode_stack


class TwoSegConcat2(ClassificationModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(TwoSegConcat2, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig300_3):
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
        comb_layer = self.combine_local_decisions_layer()
        output = comb_layer(local_decisions)
        output = tf.reduce_sum(output, axis=1, keepdims=True)
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




def main(args):
    c_log.info(__file__)
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = {
        "model_type": "distilbert-base-uncased",
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size
    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()

    task_model = TwoSegConcat2(MatrixCombine)
    def model_factory():
        # model = tf.keras.models.load_model(init_checkpoint)
        # new_model = get_regression_model()
        task_model.build_model(bert_params, model_config)
        return task_model.model

    trainer: TrainerIF = TrainerVectorRegression(
        model_config, run_config, model_factory)

    def build_dataset(input_files, is_for_training):
        return get_dummy_vector_regression_dataset(
            input_files, vocab_size, run_config, is_for_training)

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


