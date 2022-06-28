import sys

import tensorflow as tf


def file_show(fn):
    cnt = 0
    n_display = 5
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        print("---- record -----")
        for key in keys:
            if key in ["masked_lm_weights", "rel_score"]:
                v = feature[key].float_list.value
            else:
                v = feature[key].int64_list.value
            print(key)
            print(v)

        cnt += 1
        if cnt >= n_display:  ##
            break


if __name__ == "__main__":
    file_show(sys.argv[1])
