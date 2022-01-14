import tensorflow as tf


def read_tfrecord(input_files, max_seq_length):
    raw_dataset = tf.data.TFRecordDataset(input_files)
    name_to_features = {
        "input_ids" : tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.io.FixedLenFeature([1], tf.int64),
        "data_id": tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, name_to_features)

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def read_tfrecord_as_triples(input_files, max_seq_length, batch_size, shuffle):
    dataset = read_tfrecord(input_files, max_seq_length)

    def d_to_triple(e):
        return e['input_ids'], e['input_mask'], e['segment_ids']

    dataset = dataset.map(d_to_triple)
    if shuffle:
        dataset.shuffle(100)
    return dataset.batch(batch_size)


def expand_input_files(input_file_str):
    input_files = []
    for input_pattern in input_file_str.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files
