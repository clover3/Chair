import logging
import os
import pickle
import time

from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, CounterFilter
from tlm.qtype.qtype_input_fn import input_fn_builder_qtype
from tlm.qtype.qtype_model_fn import model_fn_qtype_pairwise
from tlm.training.train_flags import *


def verify_checkpoint(checkpoint_path):
    checkpoint_path_found = tf.train.latest_checkpoint(checkpoint_path)
    if not checkpoint_path_found:
        raise FileNotFoundError("Cannot find checkpoint : " + checkpoint_path)


def run_estimator(model_fn, input_fn, host_call=None):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)
    tf.io.gfile.makedirs(FLAGS.output_dir)
    if FLAGS.do_predict:
        tf_logging.addFilter(CounterFilter())

    tpu_cluster_resolver = None

    if FLAGS.use_tpu:
        raise Exception("FLAGS.use_tpu is expected to be False")

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        tf_random_seed=FLAGS.random_seed,
        train_distribute=strategy
    )
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




@report_run
def main(_):
    input_fn = input_fn_builder_qtype(FLAGS.max_seq_length, FLAGS)
    model_fn = model_fn_qtype_pairwise(FLAGS)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
