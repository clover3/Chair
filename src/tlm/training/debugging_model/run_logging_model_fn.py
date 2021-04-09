import logging

from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import MuteEnqueueFilter
from tf_util.tf_logging import tf_logging, CounterFilter
from tlm.model_cnfig import JsonConfig
from tlm.training.debugging_model.logging_model_fn import model_fn_logging_debug
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_ada
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *


def run_estimator(model_fn, input_fn, host_call=None):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)
    #FLAGS.init_checkpoint = auto_resolve_init_checkpoint(FLAGS.init_checkpoint)
    tf.io.gfile.makedirs(FLAGS.output_dir)
    if FLAGS.do_predict:
        tf_logging.addFilter(CounterFilter())

    tpu_cluster_resolver = None
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
      )

    if FLAGS.random_seed is not None:
        tf_logging.info("Using random seed : {}".format(FLAGS.random_seed))
        tf.random.set_seed(FLAGS.random_seed)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.compat.v1.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={
          'batch_size': 16
      }
    )

    if FLAGS.do_train:
        tf_logging.info("***** Running training *****")
        tf_logging.info("  Batch size = %d", FLAGS.train_batch_size)

        estimator.train(input_fn=input_fn, max_steps=FLAGS.num_train_steps)



@report_run
def main(_):
    input_files = get_input_files_from_flags(FLAGS)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    model_fn = model_fn_logging_debug(
        config,
        train_config,
    )
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())
    input_fn = input_fn_builder_ada(input_files, FLAGS, FLAGS.do_train)
    result = run_estimator(model_fn, input_fn)
    return result


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
