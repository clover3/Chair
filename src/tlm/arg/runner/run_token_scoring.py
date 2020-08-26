import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.arg.token_scoring import token_scoring_model
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.training import flags_wrapper
from tlm.training.input_fn import input_fn_token_scoring
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train token scoring")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files = flags_wrapper.get_input_files()

    input_fn = input_fn_token_scoring(input_files, FLAGS, is_training)
    model_fn = token_scoring_model(config, train_config, BertModel, "")
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
