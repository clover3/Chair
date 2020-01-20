import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.albert import BertologyFactory
from tlm.model.horizon import HorizontalAlpha
from tlm.model_cnfig import JsonConfig
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_config import TrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train horizon classification")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfig.from_flags(FLAGS)
    is_training = FLAGS.do_train
    input_files = get_input_files_from_flags(FLAGS)
    input_fn = input_fn_builder_classification(input_files, FLAGS.max_seq_length, is_training)
    model_fn = model_fn_classification(config, train_config, BertologyFactory(HorizontalAlpha))
    return run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

