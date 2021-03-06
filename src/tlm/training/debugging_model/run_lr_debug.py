from my_tf import tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, CounterFilter
from tlm.model.base import BertConfig
from tlm.training.debugging_model.lr_debug import model_fn_classification_for_lr_debug
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files, input_fn_from_flags
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


def run_w_data_id():
    input_files = get_input_files_from_flags(FLAGS)
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    model_fn = model_fn_classification_for_lr_debug(
        bert_config,
        train_config,
    )
    if FLAGS.do_predict:
        tf_logging.addFilter(CounterFilter())
    input_fn = input_fn_from_flags(input_fn_builder_classification, FLAGS)

    result = run_estimator(model_fn, input_fn)
    return result


@report_run
def main(_):
    return run_w_data_id()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
