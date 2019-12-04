import tensorflow as tf

from tf_util.tf_logging import tf_logging, logging

tf_logging.info("This is TF log")

from absl import logging as ab_logging


def main(_):
    tf_logging2 = logging.getLogger('tensorflow')
    tf_logging.setLevel(logging.INFO)
    ab_logging.info("This is ab logging")
    tf_logging.info("This is TF log")
    tf_logging2.info("TFLog 2")




if __name__ == "__main__":
    tf.compat.v1.app.run()
