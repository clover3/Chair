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


