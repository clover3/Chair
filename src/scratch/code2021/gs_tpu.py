import warnings

from tlm.training.flags_wrapper import input_fn_from_flags

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from taskman_client.wrapper import report_run
from tlm.training.input_fn import input_fn_builder_classification as input_fn_builder
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator
from tlm.training.train_config import TrainConfigEx
from models.transformer import optimization_v2 as optimization
from my_tf import tf
from tlm.training.model_fn_common import log_var_assignments


def model_fn_classification(train_config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    output_weights = tf.compat.v1.get_variable(
        "output_weights", [4, 4],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    )

    loss = tf.reduce_sum(output_weights)
    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}

    scaffold_fn = None
    log_var_assignments(tvars, initialized_variable_names)

    TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
        output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
        output_spec = NotImplemented
    else:
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={},
                scaffold_fn=scaffold_fn)

    return output_spec
  return model_fn


@report_run
def main_inner():
    train_config = TrainConfigEx.from_flags(FLAGS)
    model_fn = model_fn_classification(
        train_config=train_config,
    )
    input_fn = input_fn_from_flags(input_fn_builder, FLAGS)
    r = run_estimator(model_fn, input_fn)
    return r


def main(_):
    return main_inner()


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
