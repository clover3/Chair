from tensorflow.python.distribute.distribute_lib import Strategy

from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result
import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.custom_loop.trainer_if import EvalObjectIF


class EvalObject(EvalObjectIF):
    def __init__(self, model, eval_batches, dist_strategy: Strategy,
                 loss_fn,
                 eval_metrics,
                 eval_steps=10):
        self.loss = tf.keras.metrics.Mean(name='dev_loss')
        self.metrics: Dict[str, tf.keras.metrics.Metric] = eval_metrics
        self.eval_batches = eval_batches
        self.model = model
        self.dist_strategy: Strategy = dist_strategy
        self.loss_fn = loss_fn
        self.eval_steps = eval_steps

    @tf.function
    def eval_fn(self, item):
        x, y = item
        prediction = self.model(x, training=False)
        loss = self.loss_fn(y, prediction)
        self.loss.update_state(loss)
        for m in self.metrics.values():
            m.update_state(y, prediction)

    def do_eval(self):
        for m in self.metrics.values():
            m.reset_state()

        max_step = sum(1 for _ in self.eval_batches)

        if self.eval_steps >= 0:
            slice_step = self.eval_steps
        else:
            slice_step = max_step

        iterator = iter(self.eval_batches)
        for idx in range(slice_step):
            args = next(iterator),
            per_replica = self.dist_strategy.run(self.eval_fn, args=args)

        eval_loss = self.loss.result().numpy()
        metrics = self.metrics
        metric_res = fetch_metric_result(metrics)
        return eval_loss, metric_res
