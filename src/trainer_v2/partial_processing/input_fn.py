from typing import List

import tensorflow as tf


def create_classifier_dataset(file_path, seq_length, batch_size, is_training):
    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        return x, y

    return create_dataset_common(_select_data_from_record, batch_size, decode_record, file_path, is_training)


def create_two_seg_classification_dataset(file_path: str,
                                          seq_length_list: List[int],
                                          batch_size: int,
                                          is_training: bool):

    def decode_record(record):
        name_to_features = {
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([seq_length_list[i]], tf.int64)

            name_to_features[f'input_ids{i}'] = fixed_len_feature()
            name_to_features[f'input_mask{i}'] = fixed_len_feature()
            name_to_features[f'segment_ids{i}'] = fixed_len_feature()

        return tf.io.parse_single_example(record, name_to_features)

    def reform_example(record):
        x_list = []
        for i in range(2):
            x = {
                'input_word_ids': record[f'input_ids{i}'],
                'input_mask': record[f'input_mask{i}'],
                'input_type_ids': record[f'segment_ids{i}']
            }
            x_list.append(x)
        y = record['label_ids']
        return tuple(x_list), y

    return create_dataset_common(reform_example, batch_size, decode_record, file_path, is_training)


def create_dataset_common(select_data_from_record_fn, batch_size, decode_record, file_path, is_training):
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


def build_classification_dataset(model_config, input_files, run_config, is_training):
    dataset = create_classifier_dataset(
        tf.io.gfile.glob(input_files),
        model_config.max_seq_length,
        run_config.batch_size,
        is_training=is_training)

    return dataset