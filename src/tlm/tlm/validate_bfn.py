import os
import sys

import tensorflow as tf


def validate_data(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature

        v1 = feature["loss_base"].float_list.value
        v2 = feature["loss_target"].float_list.value
        for t1,t2 in zip(v1,v2):
            print(t1,t2, t1-t2)


if __name__ == "__main__":
    file_pathj = os.path.join("/mnt/nfs/work3/youngwookim/data/bert_tf/", sys.argv[1])
    validate_data(file_pathj)