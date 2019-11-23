import tensorflow as tf
from packaging import version


if version.parse(tf.__version__) < version.parse("1.99.9"):
    flags = tf.flags
else:
    from absl import flags

