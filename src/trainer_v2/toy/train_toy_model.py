import os
import sys

import tensorflow as tf
from official.modeling import performance

from cpath import common_model_dir_root
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_fit.modeling import get_optimizer
from trainer_v2.keras_fit.run_bert_based_classifier import run_keras_fit
from trainer_v2.run_config import RunConfigEx
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from trainer_v2.train_util.input_fn_common import create_dataset_common, get_input_fn


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
class ActuallyMLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ActuallyMLP, self).__init__()
        input_x = tf.keras.layers.Input(shape=(10,), dtype=tf.int32, name="x")
        x = tf.cast(input_x, tf.float32)
        hidden = tf.keras.layers.Dense(10, activation='relu')(x)
        predictions = tf.keras.layers.Dense(1, name="sentence_prediction")(hidden)
        inputs = [input_x]

        super(ActuallyMLP, self).__init__(
            inputs=inputs, outputs=predictions, **kwargs)


def model_factory(run_config):
    def get_model_fn():
        outer_model = ActuallyMLP()
        optimizer = get_optimizer(run_config)
        outer_model.optimizer = performance.configure_optimizer(optimizer)
        return outer_model, []
    return get_model_fn


def run_regression(args,
                   run_config,
                   get_model_fn,
                   train_input_fn,
                   eval_input_fn):

    strategy = get_strategy(args.use_tpu, args.tpu_name)
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    with strategy.scope():
        history, outer_model = run_keras_fit(get_model_fn, loss_fn, [],
                                             run_config, train_input_fn, eval_input_fn)
    stats = {'total_training_steps': run_config.steps_per_epoch * run_config.get_epochs()}

    if 'loss' in history.history:
        stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
        stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return outer_model, stats


def create_regression_dataset(file_path: str,
                           batch_size: int,
                           is_training: bool):

    def decode_record(record):
        name_to_features = {
            'y': tf.io.FixedLenFeature([], tf.int64),
            'x': tf.io.FixedLenFeature([10], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def reform_example(record):
        return record['x'], record['y']

    return create_dataset_common(reform_example, batch_size, decode_record, file_path, is_training)


def main(args):
    c_log.info("train_toy_model.py")
    output_dir = os.path.join(common_model_dir_root, "toy_model")
    num_epochs = 10
    steps_per_epoch = 5
    run_config = RunConfigEx(model_save_path=output_dir,
                             train_step=num_epochs * steps_per_epoch,
                             steps_per_execution=1,
                             steps_per_epoch=steps_per_epoch,
                             )
    get_model_fn = model_factory(run_config)
    def get_dataset(input_files, is_training):
        dataset = create_regression_dataset(input_files,
                                                        run_config.batch_size,
                                                        is_training)
        return dataset
    train_input_fn, eval_input_fn = get_input_fn(args, get_dataset)
    run_regression(args, run_config, get_model_fn, train_input_fn, eval_input_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
