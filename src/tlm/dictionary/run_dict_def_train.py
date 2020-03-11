from my_tf import tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, logging
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.seq2seq_model_fn import input_fn_builder, mask_lm_as_seq2seq
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)

    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_fn = input_fn_builder(get_input_files_from_flags(FLAGS), FLAGS, is_training)
    model_fn = mask_lm_as_seq2seq(config, train_config)

    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
