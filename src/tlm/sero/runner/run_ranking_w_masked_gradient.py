from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.model.ranking_model_weighted_loss import model_fn_ranking_w_gradient_adjust
from tlm.training.flags_wrapper import show_input_files, get_input_files_from_flags
from tlm.training.input_fn_common import format_dataset
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


def input_fn_builder_pairwise(max_seq_length, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids1":tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask1":tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids1":tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }
        name_to_features["use_pos"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


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
