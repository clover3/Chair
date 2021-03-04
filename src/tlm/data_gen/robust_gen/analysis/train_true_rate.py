import sys
from collections import Counter

import tensorflow as tf


def file_show(fn):
    cnt = 0
    counter = Counter()
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        label = feature['label_ids'].int64_list.value[0]
        counter[label] += 1
    print(counter)
##


if __name__ == "__main__":
    file_show(sys.argv[1])
