from typing import Dict

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops


from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.train_util.get_tpu_strategy import get_strategy2


def fetch_metric_result(metrics: Dict[str, tf.keras.metrics.Metric]):
    metric_res = {}
    for name, m in metrics.items():
        if isinstance(m, tf.keras.metrics.Metric):
            metric_res[name] = m.result().numpy()
        else:
            metric_res[name] = m

    if "precision" in metrics and "recall" in metrics and "f1" not in metrics:
        try:
            p = metric_res["precision"]
            r = metric_res["recall"]
            f1 = (2 * p * r) / (p + r + 1e-7)
            metrics['f1'] = f1
        except:
            pass

    return metric_res


def get_strategy_from_config(run_config: RunConfig2):
    if run_config.device_config is not None:
        device_config = run_config.device_config
        return get_strategy2(
            device_config.use_tpu,
            device_config.tpu_name,
            device_config.force_use_gpu
        )
    else:
        return get_strategy2(False, "")


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