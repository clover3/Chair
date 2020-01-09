import tensorflow as tf

import tlm.model.base as modeling
from taskman_client.wrapper import report_run
from tlm.model.base import BertModel
from tlm.tlm.loss_diff_prediction_model import loss_diff_prediction_model_online
from tlm.training.dynamic_mask_main import LMTrainConfig
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = get_input_files_from_flags(FLAGS)
    train_config = LMTrainConfig.from_flags(FLAGS)

    show_input_files(input_files)

    model_fn = loss_diff_prediction_model_online(
        bert_config=bert_config,
        train_config=train_config,
        model_class=BertModel,
    )
    if FLAGS.do_train:
        input_fn = input_fn_builder_unmasked(
            input_files=input_files,
            flags=FLAGS,
            is_training=False)
    elif FLAGS.do_eval or FLAGS.do_predict:
        input_fn = input_fn_builder_unmasked(
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
