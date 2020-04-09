import warnings

from tlm.training.flags_wrapper import input_fn_from_flags

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import tlm.model.base as modeling
from taskman_client.wrapper import report_run
from tlm.model.base import BertModel
from tlm.training.model_fn_classification_gradient import model_fn_classification
from tlm.training.input_fn import input_fn_builder_classification as input_fn_builder
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator
from tlm.training.train_config import TrainConfigEx


@report_run
def main_inner(model_class=None):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    if model_class is None:
        model_class = BertModel

    special_flags = FLAGS.special_flags.split(",")
    model_fn = model_fn_classification(
        bert_config=bert_config,
        train_config=train_config,
        model_class=model_class,
        special_flags=special_flags,
    )

    input_fn = input_fn_from_flags(input_fn_builder, FLAGS)
    r = run_estimator(model_fn, input_fn)
    return r


def main(_):
    return main_inner()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
