import os
import pickle

import tensorflow as tf

from cpath import output_path


def read_tfrecord(filepath):
    raw_dataset = tf.data.TFRecordDataset([filepath])
    max_seq_length = 300
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


def read_tfrecord_as_triples(file_path, batch_size):
    dataset = read_tfrecord(file_path)

    def d_to_triple(e):
        return e['input_ids'], e['input_mask'], e['segment_ids']

    dataset = dataset.map(d_to_triple)
    return dataset.batch(batch_size)


def main():
    file_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "bert_alamri1")
    dataset = read_tfrecord(file_path)

    dataset_numpy = list(dataset.as_numpy_iterator())
    file_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "bert_alamri1.pickle")
    pickle.dump(dataset_numpy, open(file_path, "wb"))


if __name__ == "__main__":
    main()