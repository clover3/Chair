import tensorflow as tf
from packaging import version


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