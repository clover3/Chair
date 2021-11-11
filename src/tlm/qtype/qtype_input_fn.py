import tensorflow as tf

from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn_common import format_dataset

feature_names_for_qtype = [
    "input_ids1",
    "input_mask1",
    "segment_ids1",
    "input_ids2",
    "input_mask2",
    "segment_ids2",
    "drop_input_ids1",
    "drop_input_mask1",
    "drop_segment_ids1",
    "drop_input_ids2",
    "drop_input_mask2",
    "drop_segment_ids2",
]


def input_fn_builder_qtype(max_seq_length, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {}
        for key in feature_names_for_qtype:
            name_to_features[key] = tf.io.FixedLenFeature([max_seq_length], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_qde(max_q_seq_length, max_seq_length, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    short_features = [
        "q_e_input_ids1",
        "q_e_segment_ids1",
        "q_e_input_ids2",
        "q_e_segment_ids2",
    ]
    long_features = [
        "d_e_input_ids1",
        "d_e_segment_ids1",
        "d_e_input_ids2",
        "d_e_segment_ids2",
    ]

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {}
        for key in short_features:
            name_to_features[key] = tf.io.FixedLenFeature([max_q_seq_length], tf.int64)
        for key in long_features:
            name_to_features[key] = tf.io.FixedLenFeature([max_seq_length], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_qde_score(max_q_seq_length, max_seq_length, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    short_features = [
        "q_e_input_ids",
        "q_e_segment_ids",
    ]
    long_features = [
        "d_e_input_ids",
        "d_e_segment_ids",
    ]

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {}
        for key in short_features:
            name_to_features[key] = tf.io.FixedLenFeature([max_q_seq_length], tf.int64)
        for key in long_features:
            name_to_features[key] = tf.io.FixedLenFeature([max_seq_length], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.float32)

        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
