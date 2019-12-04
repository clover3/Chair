import tensorflow as tf

from tf_util.tf_logging import tf_logging, logging
from tlm.training.dynamic_mask_main import lm_pretrain
from tlm.training.train_flags import *

flags.DEFINE_integer("predict_begin", 0, "File ")
flags.DEFINE_integer("predict_end", 100, "Number of classes (in case of classification task.")

def main(_):
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)

    tf_logging.filters[0].excludes.extend(["Dequeue next", "Enqueue next"])
    tf.io.gfile.makedirs(FLAGS.output_dir)
    tf_logging.info("Predict Runner")

    file_prefix = FLAGS.input_file

    for st in range(100, 2000, 100):
        tf_logging.info("Starting {}".format(st))
        input_files = []
        ed = st + 100
        FLAGS.out_file = "./disk_output/bert_{}_{}.pickle".format(st, ed)
        for i in range(st, ed):
            input_files.append(file_prefix + "{}".format(i))

        lm_pretrain(input_files)

if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
