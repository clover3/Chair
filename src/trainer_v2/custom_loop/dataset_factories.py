import tensorflow as tf

from trainer_v2.custom_loop.modeling_common.assymetric import ModelConfig2Seg
from trainer_v2.custom_loop.modeling_common.bert_common import ModelConfig
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.train_util.input_fn_common import create_dataset_common


def get_classification_dataset(file_path, run_config: RunConfig2, model_config: ModelConfig) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def _select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = (record['input_ids'], record['segment_ids']), record['label_ids']
        return entry
    return create_dataset_common(_select_data_from_record, run_config.common_run_config.batch_size,
                                 decode_record, file_path, run_config.is_training())


def get_two_seg_data(file_path, run_config: RunConfig2, model_config: ModelConfig2Seg) -> tf.data.Dataset:
    seq_length_list = [model_config.max_seq_length1, model_config.max_seq_length2]

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

    # def reform_example(record):
    #     x_list = []
    #     for i in range(2):
    #         x_list.append((record[f'input_ids{i}'], record[f'segment_ids{i}']))
    #     y = record['label_ids']
    #     return tuple(x_list), y

    def reform_example(record):
        for k, v in record.items():
            if v.dtype == tf.int64:
                record[k] = tf.cast(v, tf.int32)
        x = record['input_ids0'], record['segment_ids0'], record['input_ids1'], record['segment_ids1']
        y = record['label_ids']
        return x, y

    return create_dataset_common(reform_example, run_config.common_run_config.batch_size,
                                 decode_record, file_path, run_config.is_training())


def build_dataset_repeat_segs(input_files, run_config, model_config):
    dataset = get_classification_dataset(input_files, run_config, model_config)

    def repeat_record(*record):
        (input_ids, segment_ids), y = record
        return (input_ids, segment_ids, input_ids, segment_ids), y

    return dataset.map(repeat_record)


