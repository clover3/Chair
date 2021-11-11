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


def get_classification_dataset(input_files,
                                    max_seq_length,
                                    is_training,
                                    batch_size,
                                    max_pred_steps=0,
                                    num_cpu_threads=4,
                                    ):

    """The actual input function."""
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([1], tf.int64),
    }
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        d = d.shuffle(buffer_size=1000 * 1000)

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        d = d.interleave(map_func=tf.data.TFRecordDataset, cycle_length=cycle_length)
        d = d.shuffle(buffer_size=1000 * 1000)
    else:
        d = tf.data.TFRecordDataset(input_files)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        if max_pred_steps > 0:
            d = d.take(max_pred_steps)
        #d = d.repeat()

    def decode(record):
        feature = _decode_record(record, name_to_features)
        y = {}
        x = {}
        for key in feature.keys():
            if key == "label_ids":
                y = feature[key]
            else:
                x[key] = feature[key]
        return x, y
    d = d.map(decode)
    d = d.batch(batch_size=batch_size, drop_remainder=True)
    return d
