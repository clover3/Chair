import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.dual_bert_two_input import DualBertTwoInputModel
from tlm.model.dual_model_common import dual_model_prefix1
from tlm.model_cnfig import JsonConfig
from tlm.training.cls_model_fn_multiple_lr import model_fn_classification
from tlm.training.input_fn import input_fn_builder_two_inputs_w_data_id
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("DualBertTwoInputModel Two learning rate")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_builder_two_inputs_w_data_id(FLAGS)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")

    def lr_group_b(name):
        return dual_model_prefix1 in name

    lr_factor = 1 / 50
    model_fn = model_fn_classification(config,
                                       train_config,
                                       DualBertTwoInputModel,
                                       special_flags,
                                       lr_group_b,
                                       lr_factor)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

