import tensorflow as tf

import tlm.model.base as modeling
from taskman_client.wrapper import report_run
from tlm.dictionary.ssdr_model_fn import model_fn_apr_classification
from tlm.model_cnfig import JsonConfig
from tlm.training.dict_model_fn import DictRunConfig
from tlm.training.flags_wrapper import show_input_files
from tlm.training.input_fn import input_fn_builder_classification as input_fn_builder
from tlm.training.train_config import TrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    train_config = TrainConfig.from_flags(FLAGS)
    show_input_files(input_files)

    ssdr_config = JsonConfig.from_json_file(FLAGS.model_config_file)

    model_fn = model_fn_apr_classification(
        bert_config=bert_config,
        ssdr_config=ssdr_config,
        train_config=train_config,
        dict_run_config=DictRunConfig.from_flags(FLAGS),
    )
    if FLAGS.do_train:
        input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            is_training=True)
    elif FLAGS.do_eval or FLAGS.do_predict:
        input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            is_training=False)
    else:
        raise Exception()

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
