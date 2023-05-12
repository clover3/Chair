import logging
import math
import os
from typing import Tuple, Dict, Callable, List

import tensorflow as tf
from misc_lib import RecentCounter
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.evaler_if import EvalerIF
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import fetch_metric_result, get_strategy_from_config, eval_tensor, \
    summarize_metric
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase



def tf_run_eval(run_config: RunConfig2,
                evaler: EvalerIF,
                build_dataset: Callable[[str, bool], tf.data.Dataset],
                ):

    c_log.info("tf_eval_run ENTRY")
    strategy = get_strategy_from_config(run_config)
    eval_step = run_config.eval_config.eval_step
    steps_per_execution = run_config.common_run_config.steps_per_execution
    with strategy.scope():
        c_log.debug("Loading model")
        model_path = run_config.eval_config.model_save_path
        model = tf.keras.models.load_model(model_path)
        evaler.build(model)
        metrics: Dict[str, tf.keras.metrics.Metric] = evaler.get_eval_metrics()

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    eval_dataset = eval_dataset.take(eval_step)
    eval_dataset = distribute_dataset(strategy, eval_dataset)

    @tf.function
    def distributed_eval_step(iterator, steps_per_execution):
        """The step function for one training step."""
        for _ in tf.range(steps_per_execution):
            item = next(iterator)
            per_replica_losses = strategy.run(evaler.eval_fn, args=(item,))

    num_steps = sum(1 for _ in eval_dataset)
    steps_per_execution = num_steps
    c_log.info("START Evaluation")
    iterator = iter(eval_dataset)
    step = 0
    while step < num_steps:
        distributed_eval_step(iterator, steps_per_execution)
        step += steps_per_execution

    metric_res = fetch_metric_result(metrics)
    c_log.info("{}".format(metric_res))
    c_log.info("Evaluation completed ({} steps)".format(step))
    return metric_res

