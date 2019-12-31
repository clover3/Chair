import tensorflow as tf

from google_wrap.gs_wrap import auto_resolve_init_checkpoint
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, logging
from tlm.model_cnfig import JsonConfig
from tlm.sero.sero_model_fn import model_fn_sero_lm, input_fn_builder
from tlm.training.dynamic_mask_main import LMTrainConfig
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)

    tf.io.gfile.makedirs(FLAGS.output_dir)
    tf_logging.info("Train Sero")
    FLAGS.init_checkpoint = auto_resolve_init_checkpoint(FLAGS.init_checkpoint)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = LMTrainConfig.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    if FLAGS.modeling == "sero":
        input_fn = input_fn_builder(input_files, FLAGS, config, is_training)
    elif FLAGS.modeling == "bert":
        input_fn = input_fn_builder_unmasked(input_files, FLAGS, is_training)
    else:
        assert False

    model_fn = model_fn_sero_lm(config, train_config, FLAGS.modeling)

    run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("modeling")
    tf.compat.v1.app.run()
