from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.probe.probe_model_fn import model_fn_probe
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_files = get_input_files_from_flags(FLAGS)
    show_input_files(input_files)
    input_fn = input_fn_builder_classification(input_files, FLAGS.max_seq_length, FLAGS.do_train, FLAGS)
    model_fn = model_fn_probe(FLAGS)
    tf_logging.addFilter(MuteEnqueueFilter())
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
