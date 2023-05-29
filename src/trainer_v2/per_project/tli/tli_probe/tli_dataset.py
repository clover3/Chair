import tensorflow as tf

from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.run_config2 import RunConfig2


def get_tli_dataset(
        file_path,
        run_config: RunConfig2,
        model_config: ModelConfigType,
        is_for_training,
    ) -> tf.data.Dataset:

    def decode_record(record):
        name_to_features = {
        }
        def fixed_len_feature():
            return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
        name_to_features[f'input_ids'] = fixed_len_feature()
        name_to_features[f'segment_ids'] = fixed_len_feature()
        name_to_features[f'label_ids'] = tf.io.FixedLenFeature([1], tf.int64)
        tli_label_len = model_config.max_seq_length * 3
        name_to_features[f'tli_label'] = tf.io.FixedLenFeature([tli_label_len], tf.float32)

        record = tf.io.parse_single_example(record, name_to_features)
        return record

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 is_for_training)

