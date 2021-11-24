from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.qtype.qtype_prediction.qtype_prediction_input_fn import input_fn_builder_qtype_prediction
from tlm.qtype.qtype_prediction.qtype_prediction_model_fn import qtype_prediction_model_fn
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_fn = input_fn_builder_qtype_prediction(max_seq_length=FLAGS.max_seq_length, flags=FLAGS)
    model_fn = qtype_prediction_model_fn(FLAGS)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
