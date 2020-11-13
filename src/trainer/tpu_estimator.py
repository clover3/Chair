import logging
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf

from tf_util.tf_logging import tf_logging, CounterFilter
from tlm.training.train_flags import FLAGS
bert_checkpoint_path = "gs://clovertpu/training/bert_model/bert_model.ckpt"


def verify_checkpoint(checkpoint_path):
    checkpoint_path_found = tf.train.latest_checkpoint(checkpoint_path)
    if not checkpoint_path_found:
        raise FileNotFoundError("Cannot find checkpoint : " + checkpoint_path)


def run_estimator(model_fn, input_fn, host_call=None):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)
    #FLAGS.init_checkpoint = auto_resolve_init_checkpoint(FLAGS.init_checkpoint)
    tf.io.gfile.makedirs(FLAGS.output_dir)
    if FLAGS.do_predict:
        tf_logging.addFilter(CounterFilter())

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    print("FLAGS.save_checkpoints_steps", FLAGS.save_checkpoints_steps)
    config = tf.compat.v1.ConfigProto(allow_soft_placement=False,)
    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      session_config=config,
      tf_random_seed=FLAGS.random_seed,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

    if FLAGS.random_seed is not None:
        tf_logging.info("Using random seed : {}".format(FLAGS.random_seed))
        tf.random.set_seed(FLAGS.random_seed)

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

        if FLAGS.initialize_to_predict:
            checkpoint = FLAGS.init_checkpoint
        else:
            checkpoint = None
        result = estimator.evaluate(
            input_fn=input_fn,
            steps=FLAGS.max_eval_steps,
            checkpoint_path=checkpoint
        )

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          tf_logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf_logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        return result
    if FLAGS.do_predict:
        tf_logging.info("***** Running prediction *****")
        tf_logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        if not FLAGS.initialize_to_predict:
            verify_checkpoint(estimator.model_dir)
            checkpoint=None
            time.sleep(1)
        else:
            checkpoint = FLAGS.init_checkpoint

        result = estimator.predict(input_fn=input_fn,
                                   checkpoint_path=checkpoint,
                                    yield_single_examples=False)
        pickle.dump(list(result), open(FLAGS.out_file, "wb"))
        tf_logging.info("Prediction saved at {}".format(FLAGS.out_file))


