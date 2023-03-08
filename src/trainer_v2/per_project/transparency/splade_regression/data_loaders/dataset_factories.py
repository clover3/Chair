import os
from typing import List, Callable

import tensorflow as tf

from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import parse_file_path, create_dataset_common
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


def create_dataset_common_no_batch(decode_record: Callable,
                          run_config: RunConfig2,
                          file_path: str,
                          is_training_split: bool):
    do_shuffle = is_training_split and run_config.train_config.do_shuffle
    do_repeat = is_training_split
    config = run_config.dataset_config
    batch_size = run_config.common_run_config.batch_size
    if not is_training_split:
        if run_config.common_run_config.eval_batch_size is not None:
            batch_size = run_config.common_run_config.eval_batch_size
    input_files: List[str] = parse_file_path(file_path)
    if len(input_files) > 1:
        c_log.info("{} inputs files".format(len(input_files)))
    elif len(input_files) == 0:
        c_log.error("No input files found - Maybe you dont' want this ")
        raise FileNotFoundError(input_files)
    dataset = tf.data.TFRecordDataset(input_files, num_parallel_reads=len(input_files))
    if do_shuffle:
        dataset = dataset.shuffle(config.shuffle_buffer_size)
    if do_repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_vector_regression_dataset_from_batched(
        file_path,
        vector_len: int,
        run_config: RunConfig2,
        is_for_training,
) -> tf.data.Dataset:
    ragged_list_keys = ["input_ids", "attention_mask", 'y_values', 'y_indices']

    def decode_record(record):
        def get_value_types(key):
            if key == "y_values":
                return tf.float32
            else:
                return tf.int64

        name_to_features = {}
        for key in ragged_list_keys:
            name_to_features[key] = tf.io.RaggedFeature(
                value_key=key + "_flat_values", dtype=get_value_types(key),
                partitions=[tf.io.RaggedFeature.RowLengths(key + "_len_info")]
            )

        record = tf.io.parse_single_example(record, name_to_features)
        # y_indices: [B, None]
        tensor = tf.sparse.SparseTensor(
            indices=tf.expand_dims(record['y_indices'], axis=1),
            values=record['y_values'],
            dense_shape=[vector_len, ]
        )
        record['y'] = tf.sparse.to_dense(tensor)
        return record

    return create_dataset_common_no_batch(decode_record, run_config, file_path, is_for_training)


def get_vector_regression_dataset(
        file_path,
        vector_len: int,
        run_config: RunConfig2,
        is_for_training,
) -> tf.data.Dataset:
    def decode_record(record):
        name_to_features = {
            "input_ids": tf.io.RaggedFeature(tf.int64),
            "attention_mask": tf.io.RaggedFeature(tf.int64),
            "y_indices": tf.io.RaggedFeature(tf.int64),
            "y_values": tf.io.RaggedFeature(tf.float32),
        }

        record = tf.io.parse_single_example(record, name_to_features)
        tensor = tf.sparse.SparseTensor(
            indices=tf.expand_dims(record['y_indices'], axis=1),
            values=record['y_values'],
            dense_shape=[vector_len, ]
        )

        X = {
            "input_ids": record["input_ids"],
            "segment_ids": record["attention_mask"],
        }
        # Y = tf.sparse.to_dense(tensor)
        Y = tf.zeros([1], tf.float32)
        return X, Y

    return create_dataset_common(decode_record, run_config, file_path, is_for_training)


def get_dummy_vector_regression_dataset(
        file_path,
        vector_len: int,
        run_config: RunConfig2,
        is_for_training,
) -> tf.data.Dataset:
    def decode_record(record):
        X = {
            "input_ids": tf.zeros([600, ], tf.int64),
            "segment_ids": tf.zeros([600, ], tf.int64),
        }
        # Y = tf.sparse.to_dense(tensor)
        Y = tf.zeros([1], tf.float32)
        return X, Y

    return create_dataset_common(decode_record, run_config, file_path, is_for_training)


def main():
    args = flags_parser.parse_args("")
    run_config = get_run_config2(args)
    save_dir = path_join("output", "splade", "regression_tfrecord_ub")
    save_path = os.path.join(save_dir, "one.tfrecord")
    vector_len = 30522
    dataset = get_vector_regression_dataset(save_path, vector_len, run_config, False)
    print(dataset)
    for item in dataset:
        for key in item:
            print(key, item[key].shape, item[key], )

        break


if __name__ == "__main__":
    main()