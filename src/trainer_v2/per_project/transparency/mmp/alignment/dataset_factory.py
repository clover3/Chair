import tensorflow as tf
from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.definitions import ModelConfigType, ModelConfig2Seg
from trainer_v2.custom_loop.run_config2 import RunConfig2


def read_galign(
        file_path,
        run_config: RunConfig2,
        model_config: ModelConfigType,
        is_for_training,
    ) -> tf.data.Dataset:

    def decode_record(record):
        name_to_features = {
        }
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([model_config.max_seq_length], tf.int64)
            def single_int_feature():
                return tf.io.FixedLenFeature([1], tf.int64)

            name_to_features[f'input_ids{i+1}'] = fixed_len_feature()
            name_to_features[f'token_type_ids{i+1}'] = fixed_len_feature()
            name_to_features[f'q_term_mask{i+1}'] = fixed_len_feature()
            name_to_features[f'd_term_mask{i+1}'] = fixed_len_feature()
            name_to_features[f'label{i+1}'] = single_int_feature()
            name_to_features[f'is_valid{i+1}'] = single_int_feature()

        record = tf.io.parse_single_example(record, name_to_features)
        return reform_example(record)

    def reform_example(record):
        return record

    dataset = create_dataset_common(
        decode_record,
        run_config,
        file_path,
        is_for_training)

    return dataset
