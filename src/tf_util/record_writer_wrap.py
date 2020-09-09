from collections import OrderedDict
from typing import Iterable, Callable, TypeVar

import tensorflow as tf
from packaging import version

from misc_lib import TimeEstimator

if version.parse(tf.__version__) < version.parse("1.99.9"):
    pass
else:
    tf = tf.compat.v1


class RecordWriterWrap:
    def __init__(self, outfile):
        self.writer = tf.python_io.TFRecordWriter(outfile)
        self.total_written = 0

    def write_feature(self, features):
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self.writer.write(tf_example.SerializeToString())
        self.total_written += 1

    def close(self):
        self.writer.close()


A = TypeVar('A')
B = TypeVar('B')


def write_records_w_encode_fn(output_path,
                              encode: Callable[[A], OrderedDict],
                              records: Iterable[A],
                              n_items=0
                              ):
    writer = RecordWriterWrap(output_path)
    features_list: Iterable[OrderedDict] = map(encode, records)
    if n_items > 0:
        ticker = TimeEstimator(n_items)
    for e in features_list:
        writer.write_feature(e)
        if n_items > 0:
            ticker.tick()
    writer.close()


