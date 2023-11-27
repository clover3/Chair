import tensorflow as tf

import os
import sys


def decode_record(record):
    int_list_items = ["pos_input_ids", "pos_segment_ids", "neg_input_ids", "neg_segment_ids"]
    float_items = ["pos_multiplier", "pos_value_score", "pos_norm_add_factor",
                   "neg_multiplier", "neg_value_score", "neg_norm_add_factor"]
    seq_len = 16
    name_to_features = {}
    for key in int_list_items:
        name_to_features[key] = tf.io.FixedLenFeature([seq_len], tf.int64)
    for key in float_items:
        name_to_features[key] = tf.io.FixedLenFeature([1], tf.float32)
    record = tf.io.parse_single_example(record, name_to_features)
    return record


def validate_data(fn):

    dataset = tf.data.TFRecordDataset(fn)
    prev_item = None
    prev_h = None
    for item in dataset:
        try:
            h = decode_record(item)
            prev_item = item
            prev_h = h
        except Exception:
            print("prev_item", prev_item)
            print("Prev_h", prev_h)
            print("item", item)
            raise


    # print(keys)


def validate_dir(dir_path, idx_range):
    corrupt_list = []
    for i in idx_range:
        print("Check {}".format(i))
        fn = os.path.join(dir_path, str(i))
        if os.path.exists(fn):
            try:
                validate_data(fn)
            except Exception as e:
                print(e)
                corrupt_list.append(i)

        else:
            print("WARNING data {} doesn't exist".format(i))

    if corrupt_list:
        print("Corrputed _list")
    for e in corrupt_list:
        print(e)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        dir_path = sys.argv[1]
        st = int(sys.argv[2])
        ed = int(sys.argv[3])
        validate_dir(dir_path, range(st, ed))
    elif len(sys.argv) == 2:
        p = sys.argv[1]
        validate_data(p)
