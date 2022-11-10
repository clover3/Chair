import os
import time
import warnings
from typing import Tuple, Dict, Callable, List

import tensorflow as tf

from misc_lib import ceil_divide
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


# Example : src/trainer_v2/evidence_selector/runner/run_pep_server.py
class InferenceHelper:
    def __init__(self,
                 model_factory,
                 dataset_factory,
                 strategy, batch_size=16):
        self.batch_size = batch_size
        self.strategy = strategy
        self.dataset_factory = dataset_factory
        with strategy.scope():
            self.model = model_factory()

    def predict(self, payload: List):
        st = time.time()
        dataset = self.dataset_factory(payload)
        def reform(*x):
            return x,

        dataset = dataset.map(reform)
        dataset = dataset.batch(self.batch_size)

        if self.strategy is not None:
            dataset = distribute_dataset(self.strategy, dataset)
        dataset_len = len(payload)
        maybe_step = ceil_divide(dataset_len, self.batch_size)
        verbose = 1 if maybe_step > 15 else 0
        output = self.model.predict(dataset, steps=maybe_step, verbose=verbose)

        # if self.strategy is not None:
        #     output = self.strategy.reduce(output)
        ed = time.time()
        print("Took {0:2f} for {1} items".format(ed-st, len(payload)))
        return output


class InferenceHelperSimple:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size

    def enum_batch_prediction(self, dataset):
        for batch in dataset:
            x, y = batch
            output = self.model.predict_on_batch(x)
            yield x, output




