import collections

import tensorflow as tf

import cpath
import os
from misc_lib import TimeEstimator


def create_int_feature(size):
  values = [0]  * size
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def read_bert_data(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature



def convert_write(output_file, examples):
    vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
    writers = tf.python_io.TFRecordWriter(output_file)
    cnt =0
    dummy_size = 512 * 512
    tick = TimeEstimator(10*1000)
    for feature in examples:
        new_feature = collections.OrderedDict()
        for key in feature.keys():
            new_feature[key] = feature[key]

        new_feature["dummy"] = create_int_feature(dummy_size)
        tf_example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writers.write(tf_example.SerializeToString())

        tick.tick()
        if cnt > 10 * 1000:
            break
        cnt += 1

def dev():
    path = "/mnt/nfs/work3/youngwookim/data/bert_tf/tf/done/0"
    out_path = "/mnt/nfs/work3/youngwookim/data/dummy_bert_tf/0"
    data = read_bert_data(path)
    convert_write(out_path, data)

if __name__ == "__main__":
    dev()

