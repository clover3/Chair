import os
from collections import Counter

import tensorflow as tf

from arg.perspectives.baseline_bert_gen import baseline_bert_gen, baseline_bert_gen_unbal, \
    baseline_bert_gen_unbal_resplit
from cpath import data_path
from misc_lib import exist_or_mkdir


def make_train():
    dir_path = os.path.join(data_path, "perspective_bert_tfrecord")
    exist_or_mkdir(dir_path)
    baseline_bert_gen(os.path.join(dir_path, "train"), "train")
    baseline_bert_gen(os.path.join(dir_path, "dev"), "dev")


def make_train_unbal():
    dir_path = os.path.join(data_path, "perspective_bert_tfrecord2")
    exist_or_mkdir(dir_path)
    baseline_bert_gen_unbal(os.path.join(dir_path, "train"), "train")
    baseline_bert_gen_unbal(os.path.join(dir_path, "dev"), "dev")



def make_train_unbal_resplit():
    dir_path = os.path.join(data_path, "perspective_bert_tfrecord_resplit")
    exist_or_mkdir(dir_path)
    baseline_bert_gen_unbal_resplit(os.path.join(dir_path, "train"), "train")
    baseline_bert_gen_unbal_resplit(os.path.join(dir_path, "val"), "val")



def show_distribution():
    dir_path = os.path.join(data_path, "perspective_bert_tfrecord2")
    train_out_path = os.path.join(dir_path, "train")
    dev_out_path = os.path.join(dir_path, "dev")

    def count_labels(fn):
        counter = Counter()
        for record in tf.compat.v1.python_io.tf_record_iterator(fn):
            example = tf.train.Example()
            example.ParseFromString(record)
            feature = example.features.feature
            label = feature['label_ids'].int64_list.value[0]
            counter[label] += 1

        print(counter)

    count_labels(train_out_path)
    count_labels(dev_out_path)
# Output :
# Counter({0: 23850, 1: 3169})
# Counter({0: 6118, 1: 829})


if __name__ == "__main__":
    make_train_unbal_resplit()
