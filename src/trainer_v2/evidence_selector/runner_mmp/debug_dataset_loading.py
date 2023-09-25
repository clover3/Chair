import sys

import tensorflow as tf

from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.definitions import ModelConfigType, ModelConfig512_2
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config_for_predict
from trainer_v2.train_util.arg_flags import flags_parser


def get_sequence_labeling_dataset(file_path,
                                  run_config: RunConfig2,
                                  model_config: ModelConfigType,
                                  is_for_training,
                                  ) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = (record['input_ids'], record['segment_ids']), record['label_ids']
        return entry

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            # 'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([1], tf.int64),
        }
        record = tf.io.parse_single_example(record, name_to_features)
        return select_data_from_record(record)

    return create_dataset_common(decode_record, run_config,
                                 file_path, is_for_training)


def main(args):
    run_config = get_run_config_for_predict(args)
    file_path = run_config.dataset_config.train_files_path

    src_model_config = ModelConfig512_2()
    dataset = get_sequence_labeling_dataset(file_path, run_config, src_model_config, False)
    for batch in dataset:
        break


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
