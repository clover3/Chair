import argparse
import logging
import sys

from absl import logging as ab_logging

from tf_util.tf_logging import tf_logging


def main():

    tf_logging2 = logging.getLogger('tensorflow')
    tf_logging.setLevel(logging.INFO)
    ab_logging.info("This is ab logging")
    tf_logging.info("This is TF log")
    tf_logging2.info("TFLog 2")



parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--num_gpu", default=1)

#
# if __name__ == "__main__":
#     tf.compat.v1.app.run()

if __name__  == "__main__":
    args = parser.parse_args(sys.argv[1:])
    main()