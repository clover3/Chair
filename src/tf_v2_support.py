import tensorflow as tf
from packaging import version


if version.parse(tf.__version__) < version.parse("1.99.9"):
    flags = tf.flags
    placeholder = tf.placeholder
    def disable_eager_execution():
        pass
else:
    placeholder = tf.compat.v1.placeholder
    def disable_eager_execution():
        tf.compat.v1.disable_eager_execution()


    variable_scope = tf.compat.v1.variable_scope

