import sys

import tensorflow as tf


def show_checkpoint(lm_checkpoint):
    for x in tf.train.list_variables(lm_checkpoint):
        (name, var) = (x[0], x[1])
        print(name)



if __name__ == "__main__":
    if len(sys.argv) == 2:
        show_checkpoint(sys.argv[1])
