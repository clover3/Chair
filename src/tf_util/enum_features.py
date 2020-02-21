from packaging import version

from my_tf import tf


def load_record_v1(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature

def load_record_v2(fn):
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature

def load_record_v2_eager(fn):
    for record in tf.data.TFRecordDataset(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature


def get_single_record(fn):
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        return feature

if version.parse(tf.__version__) < version.parse("1.99.9"):
    load_record = load_record_v1
else:
    load_record = load_record_v2
