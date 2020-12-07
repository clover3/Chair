import sys
from collections import Counter

import tensorflow as tf


def file_show(fn):
    cnt = 0
    label_count = Counter()
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
            if key == "label_ids":
                label = v[0]
                label_count[label] += 1

        cnt += 1
    print(label_count)

if __name__ == "__main__":
    file_show(sys.argv[1])
