import tensorflow as tf

from taskman_client.wrapper import report_run
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_prediction
from tlm.training.ranking_model_fn import model_fn_rank_pred
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_files = get_input_files_from_flags(FLAGS)

    show_input_files(input_files)

    if FLAGS.do_predict:
        model_fn = model_fn_rank_pred(FLAGS)
        input_fn = input_fn_builder_prediction(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length)
    else:
        assert False

    result = run_estimator(model_fn, input_fn)
    return result



if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
