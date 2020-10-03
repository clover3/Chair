import tensorflow as tf

from arg.tf_runner.regression_w_weight import model_fn_regression
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_regression
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Regression with weigth")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_builder_regression(get_input_files_from_flags(FLAGS), FLAGS, FLAGS.do_train)

    model_fn = model_fn_regression(config, train_config)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

