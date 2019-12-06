import tensorflow as tf

import tlm.model.base as modeling
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.tlm.loss_diff_prediction_model import loss_diff_prediction_model, input_fn_builder_masked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator, TrainConfig, show_input_files


def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    train_config = TrainConfig.from_flags(FLAGS)
    model_config = JsonConfig.from_json_file(FLAGS.model_config_file)

    show_input_files(input_files)



    model_fn = loss_diff_prediction_model(
        bert_config=bert_config,
        train_config=train_config,
        model_class=BertModel,
        model_config=model_config,
    )
    if FLAGS.do_train:
        input_fn = input_fn_builder_masked(
            input_files=input_files,
            flags=FLAGS,
            is_training=True)
    elif FLAGS.do_eval or FLAGS.do_predict:
        input_fn = input_fn_builder_masked(
            input_files=input_files,
            flags=FLAGS,
            is_training=False)
    else:
        raise Exception()

    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
