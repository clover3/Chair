import argparse
import sys
from collections import Counter

import tensorflow as tf


def file_show():
    fn = sys.argv[1]
    cnt = 0
    n_check = 10000
    counter = Counter()
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()
        vp = feature["pos_multiplier"].float_list.value[0]
        vn = feature["neg_multiplier"].float_list.value[0]


        if vp > 0.01:
            counter["pos"] += 1

        if vn > 0.01:
            counter["neg"] += 1

        counter["total"] += 1

        cnt += 1
        if cnt >= n_check:  ##
            break
    print(counter)

if __name__ == "__main__":
    file_show()
