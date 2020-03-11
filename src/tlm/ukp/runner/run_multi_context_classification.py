import tensorflow as tf
from tlm.ukp.context_classification_model import MultiContextEncoder, input_fn_builder_multi_context_classification

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train multi context classification")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_builder_multi_context_classification(FLAGS.max_seq_length,
                                                             config.max_context,
                                                             config.max_context_length,
                                                             FLAGS)
    model_fn = model_fn_classification(config, train_config, MultiContextEncoder, FLAGS.special_flags.split(","))
    return run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

