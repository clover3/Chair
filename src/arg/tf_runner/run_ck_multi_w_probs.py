from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.multiple_evidence import MESecondProb
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_cppnc_multi_evidence
from tlm.training.model_fn_binary_classification import model_fn_binary_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


def run_classification_w_second_input():
    input_files = get_input_files_from_flags(FLAGS)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    model_fn = model_fn_binary_classification(
        config,
        train_config,
        MESecondProb,
    )
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())
    input_fn = input_fn_builder_cppnc_multi_evidence(FLAGS)
    result = run_estimator(model_fn, input_fn)
    return result


@report_run
def main(_):
    return run_classification_w_second_input()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
