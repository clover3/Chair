from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.base import BertConfig, BertModel
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import  input_fn_builder_prediction_w_data_id
from tlm.training.moden_fn_sensitivity import model_fn_sensitivity
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator

@report_run
def main(_):
    input_files = get_input_files_from_flags(FLAGS)
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    model_fn = model_fn_sensitivity(
        bert_config=bert_config,
        train_config=train_config,
        model_class=BertModel,
        special_flags=special_flags,
    )
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())
    input_fn = input_fn_builder_prediction_w_data_id(input_files, FLAGS.max_seq_length)
    result = run_estimator(model_fn, input_fn)
    return result


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()

