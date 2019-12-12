import tensorflow as tf

from tlm.training.train_flags import FLAGS


def get_input_files():
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files