import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.bert_with_label import BertModelWithLabel
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.lm_model_fn import model_fn_lm
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train BertModelWithLabel")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    is_training = FLAGS.do_train
    input_files = get_input_files_from_flags(FLAGS)
    input_fn = input_fn_builder_classification(input_files, FLAGS.max_seq_length, is_training, FLAGS, repeat_for_eval=True)
    model_fn = model_fn_lm(config, train_config, BertModelWithLabel,
                           get_masked_lm_output_fn=get_masked_lm_output,
                           feed_feature=True)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
