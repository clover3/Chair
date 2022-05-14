import tensorflow as tf
from official.utils.misc import keras_utils


def parse_input_files(input_file_str):
    input_files = []
    for input_pattern in input_file_str.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


def get_custom_callback(run_config):
    custom_callbacks = []
    custom_callbacks.append(
        keras_utils.TimeHistory(
            batch_size=run_config.batch_size,
            log_steps=run_config.steps_per_execution,
            logdir=run_config.model_save_path))
    return custom_callbacks