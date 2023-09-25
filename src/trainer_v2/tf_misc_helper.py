import tensorflow as tf

from cpath import output_path
from misc_lib import path_join



def get_tf_log_dir(run_name):
    return path_join(output_path, "tf_log", run_name)


def get_tf_log_dir_val(run_name):
    return path_join(output_path, "tf_log", run_name + "_val")


class SummaryWriterWrap:
    def __init__(self, run_name, writer_name="train"):
        log_dir = get_tf_log_dir(run_name)
        self.summary_writer = tf.summary.create_file_writer(log_dir, name=writer_name)
        self.step = 0

    def set_step(self, step):
        self.step = step

    def log(self, name, value):
        with self.summary_writer.as_default(step=self.step):
            tf.summary.scalar(name, value, step=self.step)
