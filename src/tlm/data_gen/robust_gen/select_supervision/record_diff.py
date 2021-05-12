import sys

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer


def get_features_from_record(r):
    example = tf.train.Example()
    example.ParseFromString(r)
    feature = example.features.feature
    return feature

def get_input_ids_from_record(r):
    f = get_features_from_record(r)
    v = f["input_ids"].int64_list.value
    return v


def file_show(fn1, fn2):
    itr1 = tf.compat.v1.python_io.tf_record_iterator(fn1)
    itr2 = tf.compat.v1.python_io.tf_record_iterator(fn2)
    for r1, r2 in zip(itr1, itr2):
        input_ids1 = get_input_ids_from_record(r1)
        input_ids2 = get_input_ids_from_record(r2)

        differ = False
        for e1, e2 in zip(input_ids1, input_ids2):
            if e1 != e2:
                differ = True
                break

        if differ:
            print(input_ids1)
            print(input_ids2)
            break


if __name__ == "__main__":
    file_show(sys.argv[1], sys.argv[2])
