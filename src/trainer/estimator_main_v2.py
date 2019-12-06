import time

import tensorflow as tf

import tlm.model.base as modeling
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.input_fn import input_fn_builder_classification as input_fn_builder
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator, TrainConfig, show_input_files


def main(_):
    begin = time.time()

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    train_config = TrainConfig.from_flags(FLAGS)

    show_input_files(input_files)

    model_fn = model_fn_classification(
        bert_config=bert_config,
        train_config=train_config,
        logging=tf_logging,
        model_class=BertModel,
    )
    if FLAGS.do_train:
        input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            is_training=True)
    if FLAGS.do_eval:
        input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False)

    run_estimator(model_fn, input_fn)
    print("Elapsed {}".format(time.time() - begin))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
