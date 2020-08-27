import os

from tlm.arg.perspective_passage import input_fn_perspective_passage, ppnc_pairwise_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Perspective passage classification")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_perspective_passage(FLAGS)
    model_fn = ppnc_pairwise_model(config, train_config, BertModel, "")
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
