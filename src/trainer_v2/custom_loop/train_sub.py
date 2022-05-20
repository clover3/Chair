import itertools
import os
from typing import Tuple, Dict, Callable

import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.RunConfig2 import RunConfig2
from trainer_v2.custom_loop.modeling_common.bert_common import is_interesting_step
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.runner_if import RunnerIF
from trainer_v2.train_util.get_tpu_strategy import get_strategy


@tf.function
def distributed_train_step(strategy, train_step_fn, dist_inputs: Tuple):
    per_replica_losses = strategy.run(train_step_fn, args=dist_inputs)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    # global_step = tf.compat.v1.train.get_or_create_global_step()
    # new_global_step = global_step + 1
    # global_step.assign(new_global_step)
    return loss


class EvalObject:
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
        pred = tf.argmax(prediction, axis=1)
        for m in self.metrics.values():
            m.update_state(y, pred)

    def do_eval(self):
        for m in self.metrics.values():
            m.reset_state()

        if self.eval_steps >= 0:
            eval_batches = itertools.islice(self.eval_batches, self.eval_steps)
        else:
            eval_batches = self.eval_batches
        for idx, e_batch in enumerate(eval_batches):
            args = (e_batch, )
            per_replica = self.dist_strategy.run(self.eval_fn, args=args)
        eval_loss = self.loss.result().numpy()
        metric_res = {}
        for name, m in self.metrics.items():
            metric_res[name] = m.result().numpy()
        return eval_loss, metric_res


def get_strategy_from_config(run_config: RunConfig2):
    if run_config.tpu_config is not None:
        return get_strategy(run_config.tpu_config.use_tpu, run_config.tpu_config.tpu_name)
    else:
        return get_strategy(False, "")


def _evaluate(tensor):
    """Returns the numpy value of a tensor."""
    if context.executing_eagerly():
        return tensor.numpy()
    return ops.get_default_session().run(tensor)


def load_model(keras_model, model_path):
    keras_model.load(model_path)


class RecentCounter:
    def __init__(self, interval):
        self.last_idx = None
        self.interval = interval

    def update_last(self, idx):
        self.last_idx = idx

    def is_over_interval(self, idx):
        if self.last_idx is None:
            self.update_last(idx)
            return True
        elapsed = idx - self.last_idx
        if elapsed < self.interval:
            return False
        else:
            self.update_last(idx)
            return True


def tf_train_run(run_config: RunConfig2,
                 runner: RunnerIF,
                 dataset_factory: Callable[[str], tf.data.Dataset]
                 ):
    c_log.debug("tf_run_inner ENTRY")
    strategy = get_strategy_from_config(run_config)
    model_save_dir: str = run_config.train_config.model_save_path

    c_log.debug("tf_run_inner initializing dataset")
    train_dataset = dataset_factory(run_config.input_file_config.train_files_path)
    eval_dataset = dataset_factory(run_config.input_file_config.eval_files_path)
    eval_batches = distribute_dataset(strategy, eval_dataset)
    dist_train_dataset = distribute_dataset(strategy, train_dataset)
    #
    c_log.debug("Building models")
    with strategy.scope():
        runner.build_model()
        model = runner.get_keras_model()
        c_log.debug("Initializing eval object")
        eval_object = EvalObject(model,
                                 eval_batches,
                                 strategy,
                                 runner.loss_fn,
                                 runner.get_eval_metrics())

    train_itr = iter(dist_train_dataset)

    def get_model_save_path():
        current_step = _evaluate(model.optimizer.iterations)
        return os.path.join(model_save_dir, "model_{}".format(current_step))

    c_log.debug("Loading checkpoints")
    runner.do_init_checkpoint(run_config.train_config.init_checkpoint)
    c_log.info("START Training")
    current_step = _evaluate(model.optimizer.iterations)
    c_log.debug("Current step = {}".format(current_step))

    def save_fn():
        model_save_path = get_model_save_path()
        model.save(model_save_path)
        c_log.info("Model saved at {}".format(model_save_path))

    @tf.function
    def distributed_train_step(dist_inputs: Tuple):
        per_replica_losses = strategy.run(runner.train_step, args=dist_inputs)
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return loss

    eval_rc = RecentCounter(run_config.train_config.eval_every_n_step)
    save_rc = RecentCounter(run_config.train_config.save_every_n_step)
    step_idx = 0
    while step_idx < run_config.train_config.train_step:
        f_do_eval = eval_rc.is_over_interval(step_idx)
        f_do_save = save_rc.is_over_interval(step_idx) and not run_config.common_run_config.is_debug_run

        if f_do_save:
            save_fn()

        current_step = _evaluate(model.optimizer.iterations)
        c_log.debug("Current step = {}".format(current_step))
        per_step_msg = "step {0}".format(step_idx)

        for _ in tf.range(run_config.common_run_config.steps_per_execution):
            batch_item = next(train_itr)
            train_loss = distributed_train_step((batch_item,))

        metrics = runner.get_train_metrics()
        msg = summarize_metric(metrics)
        per_step_msg += msg

        if f_do_eval:
            eval_loss, eval_metrics = eval_object.do_eval()
            per_step_msg += " dev: loss={0:.2f}".format(eval_loss)
            msg = summarize_metric(eval_metrics)
            per_step_msg += msg

        if f_do_eval or is_interesting_step(step_idx):
            c_log.info(per_step_msg)

        step_idx += run_config.common_run_config.steps_per_execution
    c_log.info("Training completed")
    save_fn()


def summarize_metric(metrics) -> str:
    msg = ""
    for metric_name, metric_value in metrics.items():
        msg += "{0}={1:.2f}".format(metric_name, metric_value)
    return msg


def tf_eval_run(run_config: RunConfig2,
                runner: RunnerIF,
                ):
    c_log.debug("tf_run_inner ENTRY")
    dist_strategy = get_strategy_from_config(run_config)

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = runner.build_dataset(run_config.input_file_config.eval_files_path)
    eval_batches = distribute_dataset(dist_strategy, eval_dataset)
    #
    with dist_strategy.scope():
        c_log.debug("Loading model")
        model_path = run_config.eval_config.model_save_path
        model = tf.saved_model.load(model_path)
        runner.build_model()
        runner.set_keras_model(model)
        eval_object = EvalObject(model,
                                 eval_batches,
                                 dist_strategy,
                                 runner.loss_fn,
                                 runner.get_eval_metrics(),
                                 eval_steps=-1)

    c_log.info("START Evaluation")
    eval_loss, eval_metrics = eval_object.do_eval()
    c_log.info("{}".format(eval_metrics))
    c_log.info("Training completed")
    return eval_metrics