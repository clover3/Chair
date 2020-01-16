import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.sero.sero_model_fn import model_fn_sero_lm, input_fn_builder
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train Sero")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = LMTrainConfig.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    if "sero" in FLAGS.modeling:
        total_sequence_length = config.total_sequence_length
        input_fn = input_fn_builder(input_files, total_sequence_length, FLAGS, is_training)
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
