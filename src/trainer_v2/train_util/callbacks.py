import os

import tensorflow as tf
from official.utils.misc import keras_utils
from tensorflow import keras

from trainer_v2.chair_logging import c_log
from trainer_v2.run_config import RunConfigEx


def get_custom_callback(run_config):
    custom_callbacks = []
    custom_callbacks.append(
        keras_utils.TimeHistory(
            batch_size=run_config.batch_size,
            log_steps=run_config.steps_per_execution,
            logdir=run_config.model_save_path))
    custom_callbacks.append(CustomCallback())
    return custom_callbacks


class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


class BatchEndDebugCallback(keras.callbacks.Callback):
    def __init__(self):
        super(BatchEndDebugCallback, self).__init__()
        self.cnt = 0

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.cnt += 1
        if self.cnt < 20:
            print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        elif self.cnt == 20:
            print("Callback is working!!!")


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super(CheckpointCallback, self).__init__()
        self.checkpoint_manager = checkpoint_manager

    def _do_checkpoint_save_check(self):
        step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
        self.checkpoint_manager.save(checkpoint_number=step_counter)

    def on_epoch_end(self, epoch, logs=None):
        self._do_checkpoint_save_check()

    def on_train_end(self, logs=None):
        self._do_checkpoint_save_check()

    def on_train_batch_end(self, batch, logs=None):
        ret = self._do_checkpoint_save_check()
        if ret is not None:
            c_log.info("Checkpoint saved at {}".format(ret))


def get_checkpoint_callback(model, model_dir, optimizer, run_config: RunConfigEx):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=None,
        step_counter=optimizer.iterations,
        checkpoint_interval=run_config.save_every_n_step)
    checkpoint_callback = CheckpointCallback(checkpoint_manager)
    return checkpoint_callback


def get_summary_callback(model_dir):
    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir, update_freq=1)
    return summary_callback