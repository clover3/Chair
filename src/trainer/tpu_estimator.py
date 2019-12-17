import logging
import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf

from tf_util.tf_logging import tf_logging
from tlm.training.train_flags import FLAGS


def run_estimator(model_fn, input_fn, host_call=None):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)

    tf.io.gfile.makedirs(FLAGS.output_dir)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=False,)
    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_every_n_hours =FLAGS.keep_checkpoint_every_n_hours,
      session_config=config,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.eval_batch_size,
    )

    if FLAGS.do_train:
        tf_logging.info("***** Running training *****")
        tf_logging.info("  Batch size = %d", FLAGS.train_batch_size)
        estimator.train(input_fn=input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf_logging.info("***** Running evaluation *****")
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        result = estimator.evaluate(
            input_fn=input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          tf_logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf_logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        return result
    if FLAGS.do_predict:
        tf_logging.info("***** Running evaluation *****")
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        result = estimator.predict(input_fn=input_fn, yield_single_examples=False)

        pickle.dump(list(result), open(FLAGS.out_file, "wb"))


class TrainConfig:
    def __init__(self,
                 init_checkpoint,
                 learning_rate,
                 num_train_steps,
                 num_warmup_steps,
                 use_tpu,
                 use_one_hot_embeddings,
                 num_classes,
                 iterations_per_loop,
                 checkpoint_type,
                 use_old_logits
                 ):
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.num_classes = num_classes
        self.iterations_per_loop = iterations_per_loop
        self.checkpoint_type = checkpoint_type
        self.use_old_logits = use_old_logits

    @classmethod
    def from_flags(cls, flags):
        return TrainConfig(
            flags.init_checkpoint,
            flags.learning_rate,
            flags.num_train_steps,
            flags.num_warmup_steps,
            flags.use_tpu,
            flags.use_tpu,
            flags.num_classes,
            flags.iterations_per_loop,
            flags.checkpoint_type,
            flags.use_old_logits,
        )


def show_input_files(input_files):
    tf_logging.info("*** Input Files ***")
    for idx, input_file in enumerate(input_files):
        tf_logging.info("  %s" % input_file)
        if idx > 10:
            break
    tf_logging.info("Total of %d files" % len(input_files))