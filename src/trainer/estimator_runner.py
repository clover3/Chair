from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datetime import datetime
import time
from models.transformer.hp_finetue import HP

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from models.transformer import transformer_est
from data_generator import shared_setting
from data_generator.stance import stance_detection
import pandas as pd
import path
from misc_lib import delete_if_exist
from trainer.bert_estimator_builder import *
def get_model_dir(run_id, delete_exist = True):
    run_dir = os.path.join(path.model_path, 'runs')
    save_dir = os.path.join(run_dir, run_id)
    if delete_exist:
        delete_if_exist(save_dir)
    return save_dir


# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be the name used when "
    "creating the Cloud TPU. To find out the name of TPU, either use command "
    "'gcloud compute tpus list --zone=<zone-name>', or use "
    "'ctpu status --details' if you have created your Cloud TPU using 'ctpu up'.")

# Model specific parameters
tf.flags.DEFINE_string(
    "model_dir", default="",
    help="This should be the path of storage bucket which will be used as "
    "model_directory to export the checkpoints during training.")
tf.flags.DEFINE_integer(
    "batch_size", default=128,
    help="This is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer(
    "train_steps", default=1000,
    help="Total number of training steps.")
tf.flags.DEFINE_integer(
    "eval_steps", default=4,
    help="Total number of evaluation steps. If `0`, evaluation "
    "after training is skipped.")

tf.flags.DEFINE_bool(
    "use_tpu", default=False,
    help="1 to use tpu")

tf.flags.DEFINE_string(
    "data_dir", default="",
    help="")

tf.flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

FLAGS = tf.flags.FLAGS


class EstimatorRunner:
    def __init__(self):
        self.modle = None

    @staticmethod
    def get_feature_column():
        my_feature_columns = []
        for key in ["features"]:
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        return my_feature_columns

    def stance_cold(self):
        hp = hyperparams.HPColdStart()
        topic = "atheism"
        setting = shared_setting.TopicTweets2Stance(topic)
        model_dir = get_model_dir("stance_cold_{}".format(topic))

        task = Classification(3)
        model = Transformer(hp, setting.vocab_size, task)
        param = {
            'feature_columns': self.get_feature_column(),
            'n_classes': 3,
        }
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=model_dir,
            params=param,
            config=None)

        data_source = stance_detection.DataLoader(topic, hp.seq_max, setting.vocab_filename)

        def train_input_fn(features, labels, batch_size):
            f_dict = pd.DataFrame(data=features)
            dataset = tf.data.Dataset.from_tensor_slices((f_dict, labels))
            # Shuffle, repeat, and batch the examples.
            return dataset.shuffle(1000).repeat().batch(batch_size)

        def dev_input_fn(batch_size):
            features, labels = data_source.get_dev_data()
            f_dict = pd.DataFrame(data=features)
            dataset = tf.data.Dataset.from_tensor_slices((f_dict, labels))
            # Shuffle, repeat, and batch the examples.
            return dataset.shuffle(1000).batch(batch_size)

        X, Y = data_source.get_train_data()
        num_epoch = 10
        batch_size = 32
        step_per_epoch = (len(Y)-1) / batch_size + 1
        tf.logging.info("Logging Test")
        tf.logging.info("num epoch %d", num_epoch)
        estimator.train(lambda:train_input_fn(X, Y, batch_size),
                        max_steps=num_epoch * step_per_epoch)

        print(estimator.evaluate(lambda:dev_input_fn(batch_size)))


    def train_causal(self):
        hp = hyperparams.HPCausal()
        tpu_cluster_resolver = None

        if FLAGS.use_tpu:
            model_dir = FLAGS.model_dir
            hp.batch_size = FLAGS.batch_size
            data_dir = FLAGS.data_dir
            input_pattern = os.path.join(data_dir, "Thus.train_*")
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu)
            init_checkpoint = FLAGS.init_checkpoint
        else:
            model_dir = get_model_dir("causal")
            input_pattern = os.path.join(path.data_path, "causal", "Thus.train_*")
            init_checkpoint = os.path.join(path.model_path, "runs", FLAGS.init_checkpoint)

        vocab_size = 30522

        task = Classification(3)
        model = transformer_est.TransformerEst(hp, vocab_size, task, FLAGS.use_tpu, init_checkpoint)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=model_dir,
            save_checkpoints_steps=1000,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=1000,
                num_shards=8,
                per_host_input_for_training=is_per_host))

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model.model_fn,
            config=run_config,
            train_batch_size=hp.batch_size,
            eval_batch_size=hp.batch_size)

        input_files = tf.gfile.Glob(input_pattern)
        for input_file in input_files:
            tf.logging.info("  %s" % input_file)

        train_files = input_files[1:]
        eval_files = input_files[:1]
        tf.enable_eager_execution()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", hp.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_files,
            max_seq_length=hp.seq_max,
            is_training=True)

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        class _LoggerHook(tf.train.SessionRunHook):
            def __init__(self, log_frequency):
                self.log_frequency = log_frequency

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(task.loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % self.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = self.log_frequency * 16 / duration
                    sec_per_batch = float(duration / self.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        hook = _LoggerHook(100)
        estimator.train(input_fn=train_input_fn,
                        hooks= [hook],
                        max_steps = FLAGS.train_steps
                        )

        eval_input_fn = input_fn_builder(
            input_files=eval_files,
            max_seq_length=hp.seq_max,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=20,
            )

        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))

    def train_classification(self, data_loader):
        hp = HP()
        tpu_cluster_resolver = None

        if FLAGS.use_tpu:
            model_dir = FLAGS.model_dir
            hp.batch_size = FLAGS.batch_size
            data_dir = FLAGS.data_dir
            input_pattern = os.path.join(data_dir, "Thus.train_*")
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu)
            init_checkpoint = FLAGS.init_checkpoint
        else:
            model_dir = get_model_dir("causal")
            input_pattern = os.path.join(path.data_path, "causal", "Thus.train_*")
            init_checkpoint = os.path.join(path.model_path, "runs", FLAGS.init_checkpoint)


        vocab_size = 30522

        task = Classification(3)
        model = transformer_est.TransformerEst(hp, vocab_size, task, FLAGS.use_tpu, init_checkpoint)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None,
            model_dir=model_dir,
            save_checkpoints_steps=1000,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=1000,
                num_shards=8,
                per_host_input_for_training=is_per_host))

        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model.model_fn,
            config=run_config,
            train_batch_size=hp.batch_size,
            eval_batch_size=hp.batch_size)

        input_files = tf.gfile.Glob(input_pattern)
        for input_file in input_files:
            tf.logging.info("  %s" % input_file)

        train_files = data_loader.get_train
        eval_files = input_files[:1]
        tf.enable_eager_execution()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", hp.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_files,
            max_seq_length=hp.seq_max,
            is_training=True)

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        class _LoggerHook(tf.train.SessionRunHook):
            def __init__(self, log_frequency):
                self.log_frequency = log_frequency

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(task.loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % self.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = self.log_frequency * 16 / duration
                    sec_per_batch = float(duration / self.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        hook = _LoggerHook(100)
        estimator.train(input_fn=train_input_fn,
                        hooks= [hook],
                        max_steps = FLAGS.train_steps
                        )

        eval_input_fn = input_fn_builder(
            input_files=eval_files,
            max_seq_length=hp.seq_max,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=20,
            )

        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    er = EstimatorRunner()
    er.train_causal()