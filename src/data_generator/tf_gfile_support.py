import tensorflow as tf
from packaging import version


if version.parse(tf.__version__) < version.parse("1.99.9"):
    tf_gfile = tf.gfile.GFile
else:
    tf_gfile = tf.compat.v1.gfile.GFile

