import tensorflow as tf

from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn_common import _decode_record, get_lm_basic_features, get_lm_mask_features, format_dataset


def input_fn_builder_unmasked(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "next_sentence_labels":tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_pairwise_for_bert(flags):
    return input_fn_builder_pairwise(flags.max_seq_length, flags)

def input_fn_builder_pairwise_for_sero(max_seq_length, flags):
    return input_fn_builder_pairwise(max_seq_length, flags)


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
        if flags.modeling == "multi_label_hinge":
            name_to_features["label_ids1"] = tf.io.FixedLenFeature([1], tf.int64)
            name_to_features["label_ids2"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_builder_classification(input_files,
                                         max_seq_length,
                                         is_training,
                                         flags,
                                         num_cpu_threads=4,
                                        repeat_for_eval=False):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
                "input_ids":
                        tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":
                        tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":
                        tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids":
                        tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads, repeat_for_eval)

    return input_fn


def input_fn_builder_prediction(input_files,
                                 max_seq_length,
                                 num_cpu_threads=4,):

    def input_fn(params):
        batch_size = params["batch_size"]

        name_to_features = {
                "input_ids":
                        tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":
                        tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":
                        tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }

        d = tf.data.TFRecordDataset(input_files)
        d = d.apply(
                tf.data.experimental.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        num_parallel_batches=num_cpu_threads,
                        drop_remainder=True))
        return d

    return input_fn


def input_fn_builder_masked(input_files, flags, is_training, num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        all_features = {}
        all_features.update(get_lm_basic_features(flags))
        all_features.update(get_lm_mask_features(flags))

        if flags.not_use_next_sentence:
            active_feature = ["input_ids", "input_mask", "segment_ids",
                              "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"
                              ]
        else:
            active_feature = ["input_ids", "input_mask", "segment_ids",
                              "next_sentence_labels",
                              "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"
                              ]
        selected_features = {k: all_features[k] for k in active_feature}
        return format_dataset(selected_features, batch_size, is_training, flags, input_files, num_cpu_threads)
    return input_fn


def input_fn_builder_masked2(input_files, flags, is_training, num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        all_features = {}
        all_features.update(get_lm_basic_features(flags))
        all_features.update(get_lm_mask_features(flags))

        active_feature = ["input_ids", "input_mask", "segment_ids",
                          "next_sentence_labels",
                          "masked_lm_positions", "masked_lm_ids"
                          ]
        selected_features = {k: all_features[k] for k in active_feature}
        return format_dataset(selected_features, batch_size, is_training, flags, input_files, num_cpu_threads)
    return input_fn


def input_fn_builder_blc(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "loss_valid": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "loss1": tf.io.FixedLenFeature([max_seq_length], tf.float32),
                "loss2": tf.io.FixedLenFeature([max_seq_length], tf.float32),
                "next_sentence_labels":tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_unmasked_alt_emb(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "alt_emb_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "next_sentence_labels":tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
