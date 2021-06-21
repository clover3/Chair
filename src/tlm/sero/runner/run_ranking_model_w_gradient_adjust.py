from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.model.ranking_model_weighted_loss import model_fn_ranking_w_gradient_adjust
from tlm.training.input_fn import input_fn_builder_pairwise
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    input_fn = input_fn_builder_pairwise(FLAGS.max_seq_length, FLAGS)
    model_fn = model_fn_ranking_w_gradient_adjust(FLAGS)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
