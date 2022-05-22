from typing import Dict

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def fetch_metric_result(metrics: Dict[str, tf.keras.metrics.Metric]):
    metric_res = {}
    for name, m in metrics.items():
        metric_res[name] = m.result().numpy()
    return metric_res


def get_strategy_from_config(run_config: RunConfig2):
    if run_config.tpu_config is not None:
        return get_strategy(run_config.tpu_config.use_tpu, run_config.tpu_config.tpu_name)
    else:
        return get_strategy(False, "")


def eval_tensor(tensor):
    """Returns the numpy value of a tensor."""
    if context.executing_eagerly():
        return tensor.numpy()
    return ops.get_default_session().run(tensor)


def summarize_metric(metrics: Dict[str, float]) -> str:
    msg = ""
    for metric_name, metric_value in metrics.items():
        msg += " {0}={1:.4f}".format(metric_name, metric_value)
    return msg