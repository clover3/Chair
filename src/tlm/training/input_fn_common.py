import tensorflow as tf


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(serialized=record, features=name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, dtype=tf.int32)
    example[name] = t

  return example


def get_lm_basic_features(flags):
    FixedLenFeature = tf.io.FixedLenFeature
    max_seq_length = flags.max_seq_length
    if flags.not_use_next_sentence:
        features = {
            "input_ids": FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": FixedLenFeature([max_seq_length], tf.int64),
        }
    else:
        features = {
            "input_ids": FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": FixedLenFeature([max_seq_length], tf.int64),
            "next_sentence_labels": FixedLenFeature([1], tf.int64),
        }
    return features


def get_lm_mask_features(flags):
    max_predictions_per_seq = flags.max_predictions_per_seq
    FixedLenFeature = tf.io.FixedLenFeature
    return {
        "next_sentence_labels": FixedLenFeature([1], tf.int64),
        "masked_lm_positions": FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": FixedLenFeature([max_predictions_per_seq], tf.float32),
    }


def format_dataset(name_to_features, batch_size, is_training, flags,
                   input_files,
                   num_cpu_threads,
                   repeat_for_eval=False,
                   cycle_length=250):
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        if flags.repeat_data:
            d = d.repeat()
        d = d.shuffle(buffer_size=flags.buffer_size)

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))
        cycle_length = 100
        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                block_length=flags.block_length,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=flags.buffer_size)
    else:
        d = tf.data.TFRecordDataset(input_files)

        if repeat_for_eval:
            print("repeat_for_eval",repeat_for_eval)
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



def shuffle_predict(name_to_features, batch_size, is_training, flags,
                    input_files,
                    num_cpu_threads,
                    repeat_for_eval=False,
                    cycle_length=250):
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.shuffle(buffer_size=flags.buffer_size)

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(num_cpu_threads, len(input_files))
    cycle_length = 100
    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=is_training,
            block_length=flags.block_length,
            cycle_length=cycle_length))
    d = d.shuffle(buffer_size=flags.buffer_size)
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
