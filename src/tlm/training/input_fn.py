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


def input_fn_builder_two_inputs_w_data_id(flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    max_seq_length = flags.max_seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_two_inputs_w_rel(flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    max_seq_length=flags.max_seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["rel_score"] = tf.io.FixedLenFeature([1], tf.float32)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_dual_bert_double_length_input(flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    max_seq_length=flags.max_seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids1": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask1": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids1": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn




def input_fn_builder_use_second_input(flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    max_seq_length = flags.max_seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        dataset = format_dataset(name_to_features, batch_size, is_training, flags, input_files,
                                 num_cpu_threads, False, flags.cycle_length)
        ds_renamed = dataset.map(lambda dataset: {
            'input_ids': dataset['input_ids2'],
            'segment_ids': dataset['segment_ids2'],
            'input_mask': dataset['input_mask2'],
            'label_ids': dataset['label_ids'],
            'data_id': dataset['data_id'],
        })

        return ds_renamed


    return input_fn





def input_fn_builder_cppnc_multi_evidence(flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    max_seq_length = flags.max_seq_length
    max_d_seq_length = flags.max_d_seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_d_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_d_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_d_seq_length], tf.int64),
                "input_ids3": tf.io.FixedLenFeature([max_d_seq_length], tf.int64),
                "input_mask3": tf.io.FixedLenFeature([max_d_seq_length], tf.int64),
                "segment_ids3": tf.io.FixedLenFeature([max_d_seq_length], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_builder_dot_product_ck(flags, max_sent_length, total_doc_length):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "q_input_ids": tf.io.FixedLenFeature([max_sent_length], tf.int64),
                "q_input_masks": tf.io.FixedLenFeature([max_sent_length], tf.int64),
                "c_input_ids": tf.io.FixedLenFeature([max_sent_length], tf.int64),
                "c_input_masks": tf.io.FixedLenFeature([max_sent_length], tf.int64),
                "d_input_ids": tf.io.FixedLenFeature([total_doc_length], tf.int64),
                "d_input_masks": tf.io.FixedLenFeature([total_doc_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        })
        dataset = format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)
        ds_renamed = dataset.map(lambda dataset: {
            'q_input_ids': dataset['c_input_ids'],
            'q_input_mask': dataset['c_input_masks'],
            'd_input_ids': dataset['d_input_ids'],
            'd_input_mask': dataset['d_input_masks'],
            'label_ids': dataset['label_ids'],
            'data_id': dataset['data_id'],
        })

        return ds_renamed


    return input_fn


def input_fn_builder_cppnc_triple(flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    max_seq_length = flags.max_seq_length

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids3": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask3": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids3": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_classification2(input_files,
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
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
        }
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
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            if flags.repeat_data:
                d = d.repeat()
            # d = d.shuffle(buffer_size=len(input_files))
            d = d.shuffle(buffer_size=1000 * 1000)

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                    tf.data.experimental.parallel_interleave(
                            tf.data.TFRecordDataset,
                            sloppy=is_training,
                            cycle_length=cycle_length))
            d = d.shuffle(buffer_size=1000 * 1000)
            # d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            if flags.max_pred_steps > 0:
                d = d.take(flags.max_pred_steps)
            #d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
                tf.data.experimental.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        num_parallel_batches=num_cpu_threads,
                        drop_remainder=True))
        return d

    return input_fn


def input_fn_builder_classification3(input_files,
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
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                    tf.data.experimental.parallel_interleave(
                            tf.data.TFRecordDataset,
                            sloppy=is_training,
                            cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            if flags.max_pred_steps > 0:
                d = d.take(flags.max_pred_steps)
            #d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
                tf.data.experimental.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        num_parallel_batches=num_cpu_threads,
                        drop_remainder=True))
        return d

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



def input_fn_builder_ada(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "domain_ids": tf.io.FixedLenFeature([1], tf.int64),
                "is_valid_label": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_builder_prediction_w_data_id(input_files,
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
                "data_id":
                    tf.io.FixedLenFeature([1], tf.int64),
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

def input_fn_builder_unmasked_alt_emb2(input_files,
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
                "alt_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "next_sentence_labels":tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_alt_emb2_classification(input_files,
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
                "alt_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_alt_emb_data_id_classification(input_files,
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
                "alt_input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_classification_w_focus_mask_data_id(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "focus_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_builder_classification_w_data_id(input_files,
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
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_classification_w_data_id2(input_files,
                                       max_seq_length,
                                      flags,
                                      is_training,
                                      num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_builder_classification_w_data_ids_typo(input_files,
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
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_ids": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_regression(input_files,
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
                "label_ids": tf.io.FixedLenFeature([1], tf.float32),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_builder_convert_segment_ids(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        segment_ids = tf.cast(tf.io.FixedLenFeature([max_seq_length], tf.int64) / 2, tf.int6)
        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":segment_ids,
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_builder_aux_emb_classification(input_files,
                              flags,
                              is_training,
                              dim,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "aux_emb": tf.io.FixedLenFeature([max_seq_length * dim], tf.float32),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def input_fn_token_scoring(input_files,
              flags,
              is_training,
              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([max_seq_length], tf.float32),
                "label_masks": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_token_scoring2(input_files,
              flags,
              is_training,
              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length
        label_length = max_seq_length * flags.num_classes

        name_to_features = {
                "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([label_length], tf.float32),
                "data_id": tf.io.FixedLenFeature([1], tf.int64)
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



def format_dataset_no_shuffle(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads, repeat_for_eval=False):
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        if flags.repeat_data:
            d = d.repeat()

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = 1
        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=10)
    else:
        d = tf.data.TFRecordDataset(input_files)

        if repeat_for_eval:
            d = d.repeat()

        if flags.max_pred_steps:
            n_predict = flags.eval_batch_size * flags.max_pred_steps
            d = d.take(n_predict)

        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        # d = d.repeat()
    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d


def input_fn_builder_vector_ck(flags, config):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4

    num_window = config.num_window
    max_sequence = config.max_sequence
    num_hidden = config.hidden_size

    vector_size = num_window * max_sequence * num_hidden
    mask_size = num_window * max_sequence

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "vectors": tf.io.FixedLenFeature([vector_size], tf.float32),
                "valid_mask": tf.io.FixedLenFeature([mask_size], tf.int64),
        })
        name_to_features["label_ids"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["data_id"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset_no_shuffle(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


def input_fn_query_doc(input_files,
                       flags,
                       is_training,
                       num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {
                "query": tf.io.FixedLenFeature([flags.max_query_len], tf.int64),
                "doc": tf.io.FixedLenFeature([flags.max_doc_len], tf.int64),
                "doc_mask": tf.io.FixedLenFeature([flags.max_doc_len], tf.int64),
                "label_ids": tf.io.FixedLenFeature([1], tf.int64),
                "data_id": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
