from collections import Callable

import tensorflow as tf

from trainer_v2.train_util.misc_helper import parse_input_files


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


def create_dataset_common(select_data_from_record_fn: Callable,
                          batch_size: int,
                          decode_record: Callable,
                          file_path: str,
                          is_training: bool):
    dataset = tf.data.TFRecordDataset(file_path)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        select_data_from_record_fn,
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_input_fn(args, get_dataset_fn):
    def train_input_fn():
        input_files = parse_input_files(args.input_files)
        dataset = get_dataset_fn(input_files, True)
        return dataset

    def eval_input_fn():
        if args.eval_input_files is None:
            return None
        input_files = parse_input_files(args.eval_input_files)
        dataset = get_dataset_fn(input_files, is_training=False)
        return dataset

    return train_input_fn, eval_input_fn