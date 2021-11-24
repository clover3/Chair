
import tensorflow as tf

from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn_common import format_dataset


def input_fn_builder_qtype_prediction(max_seq_length, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {}
        name_to_features["qtype_id"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["entity"] = tf.io.FixedLenFeature([max_seq_length], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
