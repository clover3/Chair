import tensorflow as tf

from arg.tf_runner.model_fn_token_scoring import model_fn_token_scoring
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_token_scoring2
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Token scoring")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_files = get_input_files_from_flags(FLAGS)
    show_input_files(input_files)
    input_fn = input_fn_token_scoring2(input_files, FLAGS, FLAGS.do_train)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")

    model_fn = model_fn_token_scoring(config, train_config)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

