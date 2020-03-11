
import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.tlm.generative_predictor import model_fn_generative_predictor
from tlm.training.flags_wrapper import input_fn_from_flags
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Run generative predictor")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_from_flags(input_fn_builder_classification, FLAGS)
    model_fn = model_fn_generative_predictor(config, train_config)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
