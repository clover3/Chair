from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.bert_hinge import HingeBert
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_classification, input_fn_builder_classification_w_data_id2
from tlm.training.model_fn_binary_classification import model_fn_binary_classification_loss
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    model_class = HingeBert
    input_files = get_input_files_from_flags(FLAGS)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    is_training = FLAGS.do_train
    model_fn = model_fn_binary_classification_loss(
        config,
        train_config,
        model_class,
    )
    if FLAGS.do_train or FLAGS.do_eval:
        input_fn = input_fn_builder_classification(input_files, FLAGS.max_seq_length, is_training, FLAGS,
                                                   num_cpu_threads=4,
                                                   repeat_for_eval=False)

    else:
        input_fn = input_fn_builder_classification_w_data_id2(
            input_files,
            FLAGS.max_seq_length,
            FLAGS,
            is_training,
            num_cpu_threads=4)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())
    result = run_estimator(model_fn, input_fn)
    return result


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()

