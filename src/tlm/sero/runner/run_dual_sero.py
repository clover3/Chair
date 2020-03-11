from functools import partial

import tensorflow as tf

from taskman_client.wrapper import report_run
from tlm.model.dual_sero_bert import DualSeroBertModel
from tlm.model_cnfig import JsonConfig
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import input_fn_from_flags
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    config = JsonConfig.from_json_file(FLAGS.bert_config_file)
    sero_config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_from_flags(input_fn_builder_classification, FLAGS)
    model_fn = model_fn_classification(config,
                                       train_config,
                                       partial(DualSeroBertModel, sero_config),
                                       FLAGS.special_flags.split(","))
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

