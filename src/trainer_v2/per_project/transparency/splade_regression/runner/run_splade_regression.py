import logging
import sys
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model, \
    get_regression_model2
from trainer_v2.per_project.transparency.splade_regression.trainer_vector_regression import TrainerVectorRegression
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import AutoTokenizer
import numpy as np


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


def main(args):
    c_log.info(__file__)
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = {
        "model_type": "distilbert-base-uncased",
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size

    def model_factory():
        # model = tf.keras.models.load_model(init_checkpoint)
        # new_model = get_regression_model(model_config)
        new_model = get_dummy_model(None)
        return new_model

    trainer: TrainerIF = TrainerVectorRegression(
        model_config, run_config, model_factory)

    def build_dataset(input_files, is_for_training):
        return get_vector_regression_dataset(
            input_files, vocab_size, run_config, is_for_training)

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


