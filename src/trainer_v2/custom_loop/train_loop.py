import logging
import math
import os
from typing import Tuple, Dict, Callable, List

import tensorflow as tf


from misc_lib import RecentCounter, ExponentialMovingAverage, SimpleMovingAverage
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result, get_strategy_from_config, eval_tensor, \
    summarize_metric
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase

from cpath import output_path
from misc_lib import path_join


@tf.function
def distributed_train_step(strategy, train_step_fn, dist_inputs: Tuple):
    per_replica_losses = strategy.run(train_step_fn, args=dist_inputs)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return loss


class ModelSaver:
    def __init__(self, model, model_save_dir, get_current_step_fn):
        self.get_current_step_fn = get_current_step_fn
        self.model_save_dir = model_save_dir
        self.model = model
        self.n_saved = 0

    def save(self):
        model_save_path = self.get_model_save_path()
        self.model.save(model_save_path)
        c_log.info("Model saved at {}".format(model_save_path))
        self.n_saved += 1

    def get_model_save_path(self):
        current_step = self.get_current_step_fn()
        return os.path.join(self.model_save_dir, "model_{}".format(current_step))


def tf_run_train(run_config: RunConfig2,
                 trainer: TrainerIFBase,
                 dataset_factory: Callable[[str, bool], tf.data.Dataset]
                 ):
    c_log.debug("tf_run_train ENTRY")
    strategy = get_strategy_from_config(run_config)
    c_log.debug("tf_run_inner initializing dataset")
    train_dataset = dataset_factory(run_config.dataset_config.train_files_path, True)
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    dist_train_dataset = distribute_dataset(strategy, train_dataset)
    eval_batches = distribute_dataset(strategy, eval_dataset)

    smooth_loss = SimpleMovingAverage(100)
    c_log.debug("Building models")
    # run a training step
    with strategy.scope():
        trainer.build_model()
        c_log.info("Loading checkpoints: {}".format(run_config.train_config.init_checkpoint))
        trainer.do_init_checkpoint(run_config.train_config.init_checkpoint)

        model = trainer.get_keras_model()
        eval_object = trainer.get_eval_object(eval_batches, strategy)

        train_itr = iter(dist_train_dataset)

        def get_current_step():
            return eval_tensor(model.optimizer.iterations)

        saver = ModelSaver(
            model, run_config.train_config.model_save_path, get_current_step
        )

        current_step = eval_tensor(model.optimizer.iterations)
        c_log.info("Current step = {}".format(current_step))

        conf_steps_per_execution = run_config.common_run_config.steps_per_execution

        @tf.function
        def distributed_train_step(train_itr, steps_per_execution):
            # try:
            total_loss = 0.0
            n_step = 0.
            for _ in tf.range(steps_per_execution):
                batch_item = next(train_itr)
                per_replica_losses = strategy.run(trainer.train_step, args=(batch_item, ))
                loss = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
                total_loss += loss
                n_step += 1.

            train_loss = total_loss / n_step
            return train_loss
        eval_rc = RecentCounter(run_config.train_config.eval_every_n_step, 0)
        save_rc = RecentCounter(run_config.train_config.save_every_n_step, 0)
        step_idx = current_step
        c_log.info("START Training")
        while step_idx < run_config.train_config.train_step:
            f_do_eval = eval_rc.is_over_interval(step_idx)
            f_do_save = save_rc.is_over_interval(step_idx) and not run_config.common_run_config.is_debug_run

            if f_do_save:
                c_log.debug("save_fn()")
                saver.save()

            current_step = eval_tensor(model.optimizer.iterations)
            c_log.debug("Current step = {}".format(current_step))

            metrics = trainer.get_train_metrics()
            for m in metrics.values():
                m.reset_state()

            if step_idx == 0:
                steps_to_execute = 1
            elif step_idx % conf_steps_per_execution > 0:
                steps_to_execute = conf_steps_per_execution - step_idx % conf_steps_per_execution
            else:
                steps_to_execute = conf_steps_per_execution
            c_log.debug("Execute {} steps".format(steps_to_execute))
            train_loss = distributed_train_step(train_itr, steps_to_execute)

            smooth_loss.update(train_loss)

            step_idx += steps_to_execute
            c_log.debug("step_idx={}".format(step_idx))
            per_step_msg = "step {0}".format(step_idx)


            trainer.train_callback()
            msg = summarize_metric(fetch_metric_result(metrics))

            per_step_msg += " loss={0:.6f} ".format(train_loss)
            per_step_msg += " loss(smooth)={0:.6f} ".format(smooth_loss.get_average())

            per_step_msg += msg

            if f_do_eval:
                eval_loss, eval_metrics = eval_object.do_eval()
                per_step_msg += " dev: loss={0:.6f}".format(eval_loss)
                msg = summarize_metric(eval_metrics)
                per_step_msg += msg

            if f_do_eval or is_interesting_step(step_idx):
                c_log.info(per_step_msg)

        c_log.info("Training completed")
        saver.save()



def tf_run_train2(run_config: RunConfig2,
                 trainer: TrainerIFBase,
                 dataset_factory: Callable[[str, bool], tf.data.Dataset]
                 ):
    c_log.debug("tf_run_train ENTRY")
    strategy = get_strategy_from_config(run_config)
    c_log.debug("tf_run_inner initializing dataset")
    train_dataset = dataset_factory(run_config.dataset_config.train_files_path, True)
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    dist_train_dataset = distribute_dataset(strategy, train_dataset)
    eval_batches = distribute_dataset(strategy, eval_dataset)
    #

    c_log.debug("Building models")
    with strategy.scope():
        trainer.build_model()
        c_log.info("Loading checkpoints: {}".format(run_config.train_config.init_checkpoint))
        trainer.do_init_checkpoint(run_config.train_config.init_checkpoint)

        model = trainer.get_keras_model()
        eval_object = trainer.get_eval_object(eval_batches, strategy)

        train_itr = iter(dist_train_dataset)

        def get_current_step():
            return eval_tensor(model.optimizer.iterations)

        saver = ModelSaver(
            model, run_config.train_config.model_save_path, get_current_step
        )

        current_step = eval_tensor(model.optimizer.iterations)
        c_log.info("Current step = {}".format(current_step))

        conf_steps_per_execution = run_config.common_run_config.steps_per_execution

        @tf.function
        def distributed_train_step(train_itr, steps_per_execution):
            # try:
            total_loss = 0.0
            n_step = 0.
            for _ in tf.range(steps_per_execution):
                batch_item = next(train_itr)
                per_replica_losses = strategy.run(trainer.train_step, args=(batch_item, ))
                loss = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_loss += loss
                n_step += 1.

            train_loss = total_loss / n_step
            return train_loss
        eval_rc = RecentCounter(run_config.train_config.eval_every_n_step, 0)
        save_rc = RecentCounter(run_config.train_config.save_every_n_step, 0)
        step_idx = current_step
        c_log.info("START Training")
        while step_idx < run_config.train_config.train_step:
            f_do_eval = eval_rc.is_over_interval(step_idx)
            f_do_save = save_rc.is_over_interval(step_idx) and not run_config.common_run_config.is_debug_run

            if f_do_save:
                c_log.debug("save_fn()")
                saver.save()

            current_step = eval_tensor(model.optimizer.iterations)
            c_log.debug("Current step = {}".format(current_step))

            metrics = trainer.get_train_metrics()
            for m in metrics.values():
                m.reset_state()

            if step_idx == 0:
                steps_to_execute = 1
            elif step_idx % conf_steps_per_execution > 0:
                steps_to_execute = conf_steps_per_execution - step_idx % conf_steps_per_execution
            else:
                steps_to_execute = conf_steps_per_execution
            c_log.debug("Execute {} steps".format(steps_to_execute))
            train_loss = distributed_train_step(train_itr, steps_to_execute)
            step_idx += steps_to_execute
            c_log.debug("step_idx={}".format(step_idx))
            per_step_msg = "step {0}".format(step_idx)

            trainer.train_callback()
            msg = summarize_metric(fetch_metric_result(metrics))
            per_step_msg += " loss={0:.6f} ".format(train_loss)
            per_step_msg += msg

            if f_do_eval:
                eval_loss, eval_metrics = eval_object.do_eval()
                per_step_msg += " dev: loss={0:.6f}".format(eval_loss)
                msg = summarize_metric(eval_metrics)
                per_step_msg += msg

            if f_do_eval or is_interesting_step(step_idx):
                c_log.info(per_step_msg)

        c_log.info("Training completed")
        saver.save()


def get_latest_model_path_from_dir_path(save_dir):
    max_name = ""
    max_step = 0
    for (dirpath, dirnames, filenames) in os.walk(save_dir):
        for dir_name in dirnames:
            prefix = "model_"
            if dir_name.startswith(prefix):
                maybe_step = dir_name[len(prefix):]
                try:
                    step = int(maybe_step)
                    if step > max_step:
                        max_step = step
                        max_name = dir_name
                except ValueError:
                    pass

        if max_name:
            return os.path.join(dirpath, max_name)
        raise FileNotFoundError("No valid file found at {}".format(save_dir))


def load_model_by_dir_or_abs(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except OSError:
        if os.path.exists(model_path):
            c_log.info("Model not found at {}, search for sub-directories".format(model_path))
            new_model_path = get_latest_model_path_from_dir_path(model_path)
            c_log.info("Loading model from {}".format(new_model_path))
            model = tf.keras.models.load_model(new_model_path)
        else:
            raise
    return model


def tf_run_eval(run_config: RunConfig2,
                trainer: TrainerIF,
                build_dataset: Callable[[str, bool], tf.data.Dataset],
                ):

    c_log.info("tf_eval_run ENTRY")
    strategy = get_strategy_from_config(run_config)
    eval_step = run_config.eval_config.eval_step
    with strategy.scope():
        c_log.debug("Loading model")
        model_path = run_config.eval_config.model_save_path
        model = tf.keras.models.load_model(model_path, compile=False)
        trainer.build_model()
        trainer.set_keras_model(model)
        loss_metric = tf.keras.metrics.Mean(name='loss')

        metrics: Dict[str, tf.keras.metrics.Metric] = trainer.get_eval_metrics()

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    eval_dataset = eval_dataset.take(eval_step)
    eval_dataset = distribute_dataset(strategy, eval_dataset)

    @tf.function
    def distributed_eval_step(iterator, steps_per_execution):
        """The step function for one training step."""
        def eval_fn(item):
            """The computation to run on each TPU device."""
            x, y = item
            prediction = model(x, training=False)
            loss = trainer.loss_fn(y, prediction)
            loss_metric.update_state(loss)
            pred = tf.argmax(prediction, axis=1)
            for m in metrics.values():
                m.update_state(y, pred)

        for _ in tf.range(steps_per_execution):
            item = next(iterator)
            per_replica_losses = strategy.run(eval_fn, args=(item,))

    num_steps = sum(1 for _ in eval_dataset)
    steps_per_execution = num_steps
    c_log.info("START Evaluation")
    iterator = iter(eval_dataset)
    step = 0
    while step < num_steps:
        distributed_eval_step(iterator, steps_per_execution)
        step += steps_per_execution

    metrics['loss'] = loss_metric
    metric_res = fetch_metric_result(metrics)
    c_log.info("{}".format(metric_res))
    c_log.info("Evaluation completed ({} steps)".format(step))
    return metric_res



def report_check(run_config: RunConfig2, ret: Dict):
    if run_config.common_run_config.report_field:
        report_field_list: List = run_config.common_run_config.report_field.split(",")
        proxy = get_task_manager_proxy()
        run_name = run_config.common_run_config.run_name
        condition = run_config.common_run_config.report_condition
        for report_field in report_field_list:
            value = float(ret[report_field])
            proxy.report_number(run_name, value, condition, report_field)
            c_log.info(f"Reported {run_name}: {report_field}={ret[report_field]} ({condition})")


def tf_run(run_config: RunConfig2,
           trainer: TrainerIF,
           build_dataset,
           ):
    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    if run_config.common_run_config.is_debug_run:
        c_log.setLevel(logging.DEBUG)

    if run_config.is_training():
        return tf_run_train(run_config, trainer, build_dataset)
    if run_config.is_eval():
        ret = tf_run_eval(run_config, trainer, build_dataset)
        report_check(run_config, ret)
        return ret



def tf_run2(
        run_config: RunConfig2,
        trainer: TrainerIFBase,
        build_dataset,
    ):
    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    if run_config.common_run_config.is_debug_run:
        c_log.setLevel(logging.DEBUG)

    if run_config.is_training():
        return tf_run_train(run_config, trainer, build_dataset)
    else:
        raise Exception("Only training is supported")


def is_interesting_step(step_idx):
    if step_idx == 0:
        return True
    interval = int(math.pow(10, int(math.log10(step_idx) - 1)))
    if step_idx < 100:
        return True
    elif step_idx % interval == 0:
        return True
    return False