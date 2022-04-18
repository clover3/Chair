import tensorflow as tf
import sys


def file_show(fn):
    for idx, record in enumerate(tf.compat.v1.python_io.tf_record_iterator(fn)):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        x_val = feature['x'].float_list.value
        y_val = feature['y'].float_list.value
        if not len(x_val) == 1769472:
            msg = "record {} len(x_val)={}".format(idx, len(x_val))
            print(msg)
            raise ValueError
        if not len(y_val) == 65536:
            msg = "record {} len(y_val)={}".format(idx, len(y_val))
            print(msg)
            raise ValueError


if __name__ == "__main__":
    file_show(sys.argv[1])
