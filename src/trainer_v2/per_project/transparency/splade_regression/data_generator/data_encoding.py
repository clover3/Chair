import os
from collections import OrderedDict
from typing import List, Dict, Any

from data_generator.create_feature import create_int_feature, create_float_feature
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import parse_file_path
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
import tensorflow as tf

from trainer_v2.train_util.arg_flags import flags_parser


def get_vector_regression_encode_fn(max_seq_length):
    def pad_truncate(items, target_len) -> List[List]:
        truncated = [t[:target_len] for t in items]
        pad_len_list = [target_len - len(t) for t in truncated]
        padded_list = [item + [0] * pad_len for item, pad_len in zip(truncated, pad_len_list)]
        return padded_list

    def encode_batched(batch: Dict[int, List]) -> OrderedDict:
        X = batch[0]
        Y = batch[1]
        input_ids, attention_mask = zip(*X)
        indices_list, values_list = zip(*Y)
        max_len = max(map(len, input_ids))
        target_len = min(max_len, max_seq_length)
        input_ids_batched = pad_truncate(input_ids, target_len)
        attention_mask_batched = pad_truncate(attention_mask, target_len)

        # flatten the list of list
        def get_ragged_list_features(ll: List[List[Any]], prefix: str):
            flat_values = []
            len_info = []
            for l in ll:
                flat_values.extend(l)
                len_info.append(len(l))

            return {
                prefix + "_flat_values": flat_values,
                prefix + "_len_info": len_info,
            }

        # ll: List of List (because batched
        ll_rep: Dict[str, List[List]] = {
            "input_ids": input_ids_batched,
            "attention_mask": attention_mask_batched,
            'y_values': values_list,
            'y_indices': indices_list,
        }

        features = OrderedDict()
        for key, ll in ll_rep.items():
            if key == 'y_indices':
                # ll will be List of List[List[int]]
                reduced_l = 0
                new_ll = []
                for l in ll:
                    assert len(l) == 1
                    reduced_l = l[0]
                    new_ll.append(reduced_l)
                ll = new_ll
            d = get_ragged_list_features(ll, key)
            for key, value in d.items():

                if type(value[0]) == int:
                    features[key] = create_int_feature(value)
                elif type(value[0]) == float:
                    features[key] = create_float_feature(value)
                else:
                    print(key, type(value[0]))
                    raise Exception()

        # Keys:
        # y_values_flat_values, y_values_len_info, y_indices_flat_values, y_indices_len_info
        return features

    return encode_batched


def get_vector_regression_dataset(
        file_path,
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
                value_key=key + "_flat_values", dtype=get_value_types(key), partitions=[
                tf.io.RaggedFeature.RowLengths(key + "_len_info")])

        record = tf.io.parse_single_example(record, name_to_features)
        return record

    input_files: List[str] = parse_file_path(file_path)
    if len(input_files) > 1:
        c_log.info("{} inputs files".format(len(input_files)))
    elif len(input_files) == 0:
        c_log.error("No input files found - Maybe you dont' want this ")
        raise FileNotFoundError(input_files)
    dataset = tf.data.TFRecordDataset(input_files, num_parallel_reads=len(input_files))
    # if do_shuffle:
    #     dataset = dataset.shuffle(config.shuffle_buffer_size)
    # if do_repeat:
    #     dataset = dataset.repeat()
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    args = flags_parser.parse_args("")
    run_config = get_run_config2(args)
    save_dir = path_join("output", "splade", "regression_tfrecord")
    save_path = os.path.join(save_dir, "all2.tfrecord")
    dataset = get_vector_regression_dataset(save_path, run_config, False)
    print(dataset)
    for item in dataset:
        for key in item:
            print(key, item[key].shape, item[key], )

        break


if __name__ == "__main__":
    main()