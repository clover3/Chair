import tensorflow as tf

import tlm.model.base as modeling
from taskman_client.wrapper import report_run
from tlm.tlm.model_fn_preserved_dim import model_fn_preserved_dim
from tlm.training.flags_wrapper import show_input_files
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    train_config = TrainConfigEx.from_flags(FLAGS)

    show_input_files(input_files)

    model_fn = model_fn_preserved_dim(
        bert_config=bert_config,
        train_config=train_config,
    )
    if FLAGS.do_predict:
        input_fn = input_fn_builder_unmasked(
            input_files=input_files,
            flags=FLAGS,
            is_training=False)
    else:
        raise Exception("Only PREDICT mode is allowed")

    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
