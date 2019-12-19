import logging
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf

from tf_util.tf_logging import tf_logging
from tlm.training.train_flags import FLAGS

def run_estimator_loop(model_fn, input_fn_list, output_name_list):
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

    if FLAGS.do_predict:
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        for input_fn, output_name in zip(input_fn_list, output_name_list):
            tf_logging.info("Predicting for %s", output_name)
            result = estimator.predict(input_fn=input_fn, yield_single_examples=False)
            pickle.dump(list(result), open(output_name, "wb"))
    else:
        raise Exception("Only predict expected")

