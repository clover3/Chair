import tensorflow as tf


def parse_input_files(input_file_str):
    input_files = []
    for input_pattern in input_file_str.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


