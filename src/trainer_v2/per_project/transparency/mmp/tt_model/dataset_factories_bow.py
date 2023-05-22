import os
from typing import List, Callable

import tensorflow as tf

from trainer_v2.custom_loop.dataset_factories import parse_file_path, create_dataset_common, create_dataset_common_inner
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import get_batch_size


def get_bow_pairwise_dataset(
        file_path,
        input_shape: InputShapeConfigTT,
        run_config: RunConfig2,
        is_for_training,
) -> tf.data.Dataset:
    max_seq_length = input_shape.max_terms * input_shape.max_subword_per_word

    def decode_record(record):
        name_to_features = {}
        for role in ["q", "d1", "d2"]:
            name_to_features[f"{role}_input_ids"] = tf.io.FixedLenFeature(max_seq_length, tf.int64)
            name_to_features[f"{role}_tfs"] = tf.io.FixedLenFeature(input_shape.max_terms, tf.int64)

        record = tf.io.parse_single_example(record, name_to_features)
        return record

    return create_dataset_common_inner(
        decode_record,
        file_path=file_path,
        do_shuffle=is_for_training,
        do_repeat=False,
        batch_size=get_batch_size(run_config, is_for_training),
        shuffle_buffer_size=run_config.dataset_config.shuffle_buffer_size,
        drop_remainder=True
    )


def get_bow_pairwise_dataset_qtw(
        file_path,
        input_shape: InputShapeConfigTT,
        run_config: RunConfig2,
        is_for_training,
) -> tf.data.Dataset:
    max_seq_length = input_shape.max_terms * input_shape.max_subword_per_word

    def decode_record(record):
        name_to_features = {}
        for role in ["q", "d1", "d2"]:
            name_to_features[f"{role}_input_ids"] = tf.io.FixedLenFeature(max_seq_length, tf.int64)
            name_to_features[f"{role}_tfs"] = tf.io.FixedLenFeature(input_shape.max_terms, tf.int64)
            name_to_features[f"{role}_qtw"] = tf.io.FixedLenFeature(input_shape.max_terms, tf.float32)

        record = tf.io.parse_single_example(record, name_to_features)
        return record

    return create_dataset_common_inner(
        decode_record,
        file_path=file_path,
        do_shuffle=is_for_training,
        do_repeat=is_for_training,
        batch_size=get_batch_size(run_config, is_for_training),
        shuffle_buffer_size=run_config.dataset_config.shuffle_buffer_size,
        drop_remainder=True
    )
