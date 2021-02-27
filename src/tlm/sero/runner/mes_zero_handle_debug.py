from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.mes_sel_var import MES_const_0_handle
from tlm.model_cnfig import JsonConfig
from tlm.sero.mes_mask_debug_model import model_fn_binary_classification_loss
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
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
        MES_const_0_handle,
    )
    input_fn = input_fn_builder_classification(input_files, FLAGS.max_d_seq_length, is_training, FLAGS,
                                               num_cpu_threads=4,
                                               repeat_for_eval=False)

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

