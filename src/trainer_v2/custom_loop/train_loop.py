import logging
import os
import warnings
from typing import Tuple, Dict, Callable, List

import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy

from cpath import get_bert_config_path
from misc_lib import RecentCounter
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log, IgnoreFilter
from trainer_v2.custom_loop.modeling_common.bert_common import is_interesting_step, load_bert_config
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result, get_strategy_from_config, eval_tensor, \
    summarize_metric
from trainer_v2.custom_loop.trainer_if import TrainerIF


@tf.function
def distributed_train_step(strategy, train_step_fn, dist_inputs: Tuple):
    per_replica_losses = strategy.run(train_step_fn, args=dist_inputs)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
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

        max_step = sum(1 for _ in self.eval_batches)

        if self.eval_steps >= 0:
            slice_step = self.eval_steps
        else:
            slice_step = max_step

        # print("Max_step {}".format(max_step))
        # eval_batches = itertools.islice(self.eval_batches, slice_step)
        # print("type(eval_batches)", type(eval_batches))
        # cnt = 0
        # for idx, e_batch in enumerate(eval_batches):
        #     args = (e_batch, )
        #     per_replica = self.dist_strategy.run(self.eval_fn, args=args)
        #     cnt += 1
        # print("do_eval {} steps".format(cnt))
        iterator = iter(self.eval_batches)
        for idx in range(slice_step):
            args = next(iterator),
            per_replica = self.dist_strategy.run(self.eval_fn, args=args)

        eval_loss = self.loss.result().numpy()
        metrics = self.metrics
        metric_res = fetch_metric_result(metrics)
        return eval_loss, metric_res


def tf_run_train(run_config: RunConfig2,
                 trainer: TrainerIF,
                 dataset_factory: Callable[[str, bool], tf.data.Dataset]
                 ):
    c_log.debug("tf_run_train ENTRY")
    strategy = get_strategy_from_config(run_config)
    model_save_dir: str = run_config.train_config.model_save_path

    c_log.debug("tf_run_inner initializing dataset")
    train_dataset = dataset_factory(run_config.dataset_config.train_files_path, True)
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    eval_batches = distribute_dataset(strategy, eval_dataset)
    dist_train_dataset = distribute_dataset(strategy, train_dataset)
    #
    c_log.debug("Building models")
    with strategy.scope():
        trainer.build_model()
        model = trainer.get_keras_model()
        c_log.debug("Initializing eval object")
        eval_object = EvalObject(model,
                                 eval_batches,
                                 strategy,
                                 trainer.loss_fn,
                                 trainer.get_eval_metrics())

        train_itr = iter(dist_train_dataset)

        def get_model_save_path():
            current_step = eval_tensor(model.optimizer.iterations)
            return os.path.join(model_save_dir, "model_{}".format(current_step))

        c_log.info("Loading checkpoints: {}".format(run_config.train_config.init_checkpoint))
        trainer.do_init_checkpoint(run_config.train_config.init_checkpoint)
        current_step = eval_tensor(model.optimizer.iterations)
        c_log.info("Current step = {}".format(current_step))

        def save_fn():
            model_save_path = get_model_save_path()
            model.save(model_save_path)
            c_log.info("Model saved at {}".format(model_save_path))

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
                save_fn()

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
            c_log.debug("step_idx={} optimizer_iter={}".format(step_idx, model.optimizer.iterations))
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
        save_fn()


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
    steps_per_execution = run_config.common_run_config.steps_per_execution
    with strategy.scope():
        c_log.debug("Loading model")
        model_path = run_config.eval_config.model_save_path
        model = tf.saved_model.load(model_path)
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


def adjust_logging():
    msgs = [
        # "UserWarning: `layer.apply` is deprecated and will be removed in a future version",
        "`model.compile_metrics` will be empty until you train or evaluate the model."
    ]
    tf_logging = logging.getLogger("tensorflow")
    tf_logging.addFilter(IgnoreFilter(msgs))
    warnings.filterwarnings("ignore", '`layer.updates` will be removed in a future version. ')
    # warnings.filterwarnings("ignore", "`layer.apply` is deprecated")


def tf_run_for_bert(dataset_factory, model_config,
                    run_config: RunConfig2, inner):
    adjust_logging()
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    trainer = Trainer(bert_params, model_config, run_config, inner)
    tf_run(run_config, trainer, dataset_factory)