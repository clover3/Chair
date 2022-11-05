import os
import warnings
from typing import Tuple, Dict, Callable, List

import tensorflow as tf
from trainer_v2.chair_logging import c_log, IgnoreFilter
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config


def tf_run_predict(run_config: RunConfig2,
                   build_dataset: Callable[[str, bool], tf.data.Dataset],
                   ):

    c_log.info("tf_eval_run ENTRY")
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        c_log.debug("Loading model")
        model_path = run_config.eval_config.model_save_path
        model = tf.saved_model.load(model_path)

    c_log.debug("tf_run_inner initializing dataset")
    eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    eval_dataset = distribute_dataset(strategy, eval_dataset)
    prediction = model.predict(eval_dataset, training=False)
    return prediction


class InferenceHelper:
    def __init__(self, model, strategy, batch_size=16):
        self.model = model
        self.batch_size = batch_size
        self.strategy = strategy

    def predict(self, dataset):
        if self.strategy is not None:
            dataset = distribute_dataset(self.strategy, dataset)
        model = self.model
        output = model.predict(dataset)
        if self.strategy is not None:
            output = self.strategy.reduce(output)
        return output

    def enum_batch_prediction(self, dataset):
        if self.strategy is not None:
            dataset = distribute_dataset(self.strategy, dataset)

        @tf.function
        def dist_pred(batch):
            return self.model(batch)

        for batch in dataset:
            x, y = batch
            output = self.strategy.run(dist_pred, (x,))
            x = self.strategy.gather(x, axis=0)
            yield x, output


class InferenceHelperSimple:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size

    def enum_batch_prediction(self, dataset):
        for batch in dataset:
            x, y = batch
            output = self.model.predict_on_batch(x)
            yield x, output
