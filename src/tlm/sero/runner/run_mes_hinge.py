from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import MuteEnqueueFilter, tf_logging
from tlm.model.mes_hinge import MES_hinge, MES_pred
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import show_input_files, get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_pairwise, input_fn_builder_classification_w_data_id2
from tlm.training.model_fn_light import model_fn_with_loss
from tlm.training.ranking_model_fn import model_fn_ranking
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
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
    if FLAGS.do_train or FLAGS.do_eval:
        model_fn = model_fn_with_loss(
            config,
            train_config,
            MES_hinge,
        )
        input_fn = input_fn_builder_pairwise(FLAGS.max_d_seq_length, FLAGS)

    else:
        model_fn = model_fn_with_loss(
            config,
            train_config,
            MES_pred,
        )

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
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
