import sys

import tensorflow as tf


def file_show(fn):
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        for key in keys:
            if key == "masked_lm_weights":
                v = feature[key].float_list.value
            else:
                v = feature[key].int64_list.value
            print(key, v)
        break

if __name__ == "__main__":
    file_show(sys.argv[1])
