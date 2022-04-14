import tensorflow as tf


def _decode_record(record, name_to_features):
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)
    convert_int64_to_32(example)
    return example


def convert_int64_to_32(example):
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = t


def format_dataset(name_to_features,
                   is_training,
                   input_files,
                   repeat_data=True,
                   buffer_size=100,
                   repeat_for_eval=False,
                   cycle_length=250):
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        if repeat_data:
            d = d.repeat()
        d = d.shuffle(buffer_size=buffer_size)
        d = d.interleave(map_func=tf.data.TFRecordDataset, cycle_length=cycle_length)
        d = d.shuffle(buffer_size=buffer_size)
    else:
        d = tf.data.TFRecordDataset(input_files)
        if repeat_for_eval:
            d = d.repeat()
    d = d.map(lambda record: _decode_record(record, name_to_features))
    return d