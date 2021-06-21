from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.mes_hinge import MES_hinge_pred_with_layer1
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_classification_w_data_id2
from tlm.training.model_fn_binary_classification import model_fn_binary_classification_loss
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


def run_mes_hinge_var():
    input_files = get_input_files_from_flags(FLAGS)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    model_fn = model_fn_binary_classification_loss(
        config,
        train_config,
        MES_hinge_pred_with_layer1,
    )
    is_training = FLAGS.do_train
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