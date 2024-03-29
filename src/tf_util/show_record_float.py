import sys

import tensorflow as tf


def file_show(fn):
    cnt = 0
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        print("---- record -----")
        for key in keys:
            print(key)
            print(feature[key].int64_list.value)
            print(feature[key].float_list.value)

        cnt += 1
        if cnt >= 5:
            break


if __name__ == "__main__":
    file_show(sys.argv[1])
