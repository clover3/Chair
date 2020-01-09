import tensorflow as tf

from tf_util.tf_logging import tf_logging
from tlm.training.train_flags import FLAGS


def get_input_files():
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


def get_input_files_from_flags(flags):
    input_files = []
    for input_pattern in flags.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


def input_fn_from_flags(input_fn_builder, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)

    return input_fn_builder(input_files, flags.max_seq_length, flags.do_train)


def show_input_files(input_files):
    tf_logging.info("*** Input Files ***")
    for idx, input_file in enumerate(input_files):
        tf_logging.info("  %s" % input_file)
        if idx > 10:
            break
    tf_logging.info("Total of %d files" % len(input_files))